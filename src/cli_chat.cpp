#include "cli.h"

#include <sentencepiece_processor.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <io.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "gguf_writer.h"
#include "json_util.h"
#include "llama_engine.h"
#include "llama_prompt.h"
#include "model.h"
#include "path_util.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

constexpr size_t kMaxContextChars = 4000;

std::string read_all_stdin() {
  std::ostringstream ss;
  ss << std::cin.rdbuf();
  return ss.str();
}

std::string read_file_text(const fs::path& p, bool* ok) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    if (ok) *ok = false;
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  if (ok) *ok = true;
  return ss.str();
}

bool write_file_text(const fs::path& p, const std::string& text) {
  std::error_code ec;
  if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f << text;
  return f.good();
}

// 找与词表大小匹配的 spm.model（模型目录优先，其次项目内候选）
fs::path find_matching_spm(const fs::path& model_dir, int64_t vocab,
                           const fs::path& explicit_spm) {
  auto matches = [vocab](const fs::path& p) {
    if (p.empty() || !fs::exists(p)) return false;
    sentencepiece::SentencePieceProcessor sp;
    if (!sp.Load(path_to_ansi(p)).ok()) return false;
    return (int64_t)sp.GetPieceSize() == vocab;
  };
  if (!explicit_spm.empty()) return matches(explicit_spm) ? explicit_spm : fs::path();
  for (const auto& c : find_spm_candidates(model_dir)) {
    if (matches(c)) return c;
  }
  return {};
}

// 路径 → 显示用 UTF-8
std::string u8(const fs::path& p) { return path_to_utf8(p); }

// ---------------------------------------------------------------------------
// llama.cpp 会话（一次性 / 交互式共用）
// ---------------------------------------------------------------------------
struct LlamaSession {
  LlamaEngine engine;
  LlamaEngineOptions opts;
  LlamaSamplingParams sampling;
  std::string system_prompt;
  bool template_mode = false;
  bool stream = true;
  std::string context;
  // 提示词通道：auto=自动（本项目模型 token，对话模型 chat 模板）
  LlamaPromptMode mode = LlamaPromptMode::kAuto;
  // 本项目导出的 GGUF 需要 sentencepiece token 通道（否则拉丁字符会 400）
  std::shared_ptr<sentencepiece::SentencePieceProcessor> sp;
  int64_t max_seq_len = 2048;

  // 是否会走对话模板通道（auto 时判断：没有匹配 spm 且模型自带 chat_template）
  bool chat_channel() const {
    if (mode == LlamaPromptMode::kChat) return true;
    if (mode != LlamaPromptMode::kAuto) return false;
    return sp == nullptr && engine.has_chat_template();
  }

  std::string compose(const std::string& user, bool include_system) const {
    std::string p;
    if (include_system && !system_prompt.empty()) p += system_prompt + "\n";
    p += context;
    if (template_mode) {
      p += "问：" + user + "\n答：";
    } else {
      p += user;
    }
    return p;
  }

  void append_turn(const std::string& user, const std::string& answer) {
    if (template_mode) {
      context += "问：" + user + "\n答：" + answer + "\n";
    } else {
      context += user + "\n" + answer + "\n";
    }
    if (context.size() > kMaxContextChars) {
      context = context.substr(context.size() - kMaxContextChars);
    }
  }

  bool ask(const std::string& user, std::string* out, std::string* err) {
    const bool chat = chat_channel();
    std::string prompt = compose(user, !chat);
    std::string text;
    const bool ok = llama_complete_smart(
        engine, prompt, sp.get(), max_seq_len, sampling,
        [this](const std::string& piece) {
          if (stream) {
            std::cout << piece << std::flush;
          }
        },
        &text, err, mode, system_prompt);
    if (ok) {
      *out = text;
      append_turn(user, text);
    }
    return ok;
  }
};

// ---------------------------------------------------------------------------
// 内置 libtorch 引擎会话
// ---------------------------------------------------------------------------
struct BuiltinSession {
  GPT model{nullptr};
  std::shared_ptr<sentencepiece::SentencePieceProcessor> sp;
  double temperature = 0.8;
  int top_k = 40;
  int max_tokens = 128;
  int64_t max_seq_len = 256;
  bool template_mode = false;
  std::string context;

  std::string compose(const std::string& user) const {
    std::string p = context;
    if (template_mode) {
      p += "问：" + user + "\n答：";
    } else if (!p.empty()) {
      p += "\n" + user;
    } else {
      p += user;
    }
    return p;
  }

  void append_turn(const std::string& user, const std::string& answer) {
    if (template_mode) {
      context += "问：" + user + "\n答：" + answer + "\n";
    } else {
      context += user + "\n" + answer + "\n";
    }
    if (context.size() > kMaxContextChars) {
      context = context.substr(context.size() - kMaxContextChars);
    }
  }

  bool ask(const std::string& user, std::string* out, std::string* err) {
    std::string full = compose(user);
    std::vector<int> ids;
    const auto status = sp->Encode(full, &ids);
    if (!status.ok()) {
      if (err) *err = "编码失败: " + status.ToString();
      return false;
    }
    if ((int64_t)ids.size() > max_seq_len) {
      const size_t drop = ids.size() - (size_t)max_seq_len;
      ids.erase(ids.begin(), ids.begin() + drop);
    }
    std::vector<int64_t> in_ids(ids.begin(), ids.end());
    std::vector<int64_t> out_ids;
    try {
      torch::NoGradGuard guard;
      const torch::Device dev = torch::cuda::is_available()
                                    ? torch::Device(torch::kCUDA)
                                    : torch::Device(torch::kCPU);
      auto input = torch::tensor(in_ids, torch::TensorOptions().dtype(torch::kInt64))
                       .reshape({1, (int64_t)in_ids.size()})
                       .to(dev);
      const std::optional<int64_t> topk =
          top_k > 0 ? std::optional<int64_t>(top_k) : std::nullopt;
      auto output = model->generate(input, max_tokens, temperature, topk);
      auto cpu = output[0].to(torch::kCPU).contiguous();
      const int64_t* d = cpu.data_ptr<int64_t>();
      out_ids.assign(d, d + cpu.numel());
    } catch (const std::exception& e) {
      if (err) *err = e.what();
      return false;
    }
    if ((int64_t)out_ids.size() <= (int64_t)ids.size()) {
      *out = std::string();
      return true;
    }
    std::vector<int> new_ids(out_ids.begin() + ids.size(), out_ids.end());
    std::string text;
    const auto s2 = sp->Decode(new_ids, &text);
    if (!s2.ok()) {
      if (err) *err = "解码失败: " + s2.ToString();
      return false;
    }
    *out = text;
    append_turn(user, text);
    return true;
  }
};

} // namespace

// ===========================================================================
// chat 帮助
// ===========================================================================
void cli_print_chat_help() {
  std::cout <<
      "用法: gpt_trainer.exe chat [选项] [--] [提示词]\n"
      "\n"
      "模型：\n"
      "  --model <路径>           saves/modelN 或 *.gguf（也可以用位置参数给出）\n"
      "  --engine <自动|llama|builtin>\n"
      "                           推理引擎（默认 auto：优先 llama.cpp）\n"
      "  --server <llama-server.exe>   llama.cpp 引擎路径（默认自动探测）\n"
      "  --spm <spm.model>        内置引擎使用的分词器（默认自动匹配词表）\n"
      "\n"
      "输入：\n"
      "  --prompt <文本>          提示词（位置参数亦可；\"-\" 表示从标准输入读取）\n"
      "  --prompt-file <文件>     从文件读取提示词\n"
      "  --interactive, --repl    交互式多轮对话（输入 exit 退出）\n"
      "  --system <文本>          系统提示词（拼在对话最前面）\n"
      "  --context <文件>         载入之前的对话上下文\n"
      "  --save-context <文件>    结束后把上下文写入文件\n"
      "  --template               按「问：… 答：」拼接（问答式模型）\n"
      "  --chat-mode <模式>       auto|chat|text|tokens：提示词通道\n"
      "                           auto=自动（本项目模型走 spm token，Qwen 等对话模型\n"
      "                           自动套用模型自带 chat 模板，其余走文本通道）\n"
      "  --no-context             单轮：不携带历史\n"
      "\n"
      "采样：\n"
      "  --max-tokens <n>         最多生成 token 数（默认 128）\n"
      "  --temperature <f>        温度（默认 0.8）\n"
      "  --top-k <n>              Top-K（默认 40）\n"
      "  --top-p <f>              Top-P（默认 0.95）\n"
      "  --repeat <f>             重复惩罚（默认 1.1）\n"
      "  --seed <n>               随机种子（默认 -1 随机）\n"
      "  --no-stream              不逐字输出（等全部生成完再打印）\n"
      "\n"
      "引擎参数（llama.cpp）：\n"
      "  --ctx <n>                上下文长度（默认 2048）\n"
      "  --threads <n>            CPU 线程数（默认 0=自动）\n"
      "  --ngl <n>                GPU 层数（默认 99，0=纯 CPU）\n"
      "\n"
      "示例：\n"
      "  gpt_trainer.exe chat --model saves/model1 --prompt \"今天天气\"\n"
      "  gpt_trainer.exe chat --model models/gguf/model1.gguf --interactive --template\n"
      "  echo 机器学习是一种 | gpt_trainer.exe chat -m saves/model1 --max-tokens 80\n"
      "  gpt_trainer.exe chat --engine builtin --model saves/model1 --prompt 你好\n";
}

void cli_print_selftest_help() {
  std::cout <<
      "用法: gpt_trainer.exe selftest [选项]\n"
      "\n"
      "  --model <目录>   待测模型目录（默认 ../py训练器/saves/model1）\n"
      "  --spm <文件>     分词器（默认 ../py训练器/models/spm.model）\n"
      "\n"
      "依次执行：加载模型 → CUDA 前向 → 生成文本 → 保存/加载回环。\n";
}

void cli_print_loadtest_help() {
  std::cout <<
      "用法: gpt_trainer.exe loadtest <文件.pt>\n"
      "\n"
      "加载任意 state_dict 文件并打印张量名称/形状/求和，用于排查互读问题。\n";
}

// ===========================================================================
// chat
// ===========================================================================
int cli_cmd_chat(const CliArgs& a, bool legacy) {
  if (a.has(L"help")) {
    cli_print_chat_help();
    return 0;
  }

  // ---- 模型 ----
  std::wstring model_w = a.get(L"model");
  std::wstring prompt_w = a.get(L"prompt");
  const std::wstring prompt_file = a.get(L"prompt-file");
  std::wstring server_w = a.get(L"server");
  std::wstring spm_w = a.get(L"spm");
  if (legacy) {
    // 旧式：--llama-chat <gguf> <提示词> [llama-server.exe]
    if (model_w.empty()) model_w = a.pos(0);
    if (prompt_w.empty()) prompt_w = a.pos(1);
    if (server_w.empty()) server_w = a.pos(2);
  } else if (model_w.empty()) {
    model_w = a.pos(0);
  }
  // 位置参数：--model 已给时第一个位置参数是提示词；否则第二个位置参数是提示词
  if (!legacy && prompt_w.empty() && !a.has(L"prompt")) {
    const size_t prompt_index = a.has(L"model") ? 0 : 1;
    if (a.positional.size() > prompt_index) prompt_w = a.positional[prompt_index];
  }
  if (model_w.empty()) {
    cli_error("请用 --model 指定模型（saves/modelN 或 *.gguf），可先运行 list 查看");
    return 2;
  }

  const fs::path model_path(model_w);
  const bool is_gguf = model_path.extension() == L".gguf";
  if (!fs::exists(model_path)) {
    cli_error("模型不存在: " + u8(model_path));
    return 2;
  }

  // ---- 引擎选择 ----
  std::wstring engine_mode = a.get(L"engine", legacy ? L"llama" : L"auto");
  if (engine_mode.empty()) engine_mode = L"auto";
  bool want_llama = false;
  if (engine_mode == L"builtin") {
    want_llama = false;
  } else if (engine_mode == L"llama") {
    want_llama = true;
  } else {
    // auto：GGUF 一定走 llama.cpp；训练模型优先 llama.cpp（导 GGUF），
    // 找不到 llama-server 时回退内置引擎
    if (is_gguf) {
      want_llama = true;
    } else {
      const fs::path server =
          server_w.empty() ? LlamaEngine::detect_server_exe() : fs::path(server_w);
      want_llama = !server.empty();
      if (!want_llama) {
        cli_say("[提示] 未找到 llama-server.exe，改用内置 libtorch 引擎"
                "（可用 --server 指定，或 --engine builtin 静默使用）");
      }
    }
  }
  if (want_llama && !is_gguf && !a.has(L"spm") && engine_mode == L"builtin") {
    want_llama = false;
  }

  // ---- 会话参数 ----
  bool interactive = a.has(L"interactive") || a.has(L"repl");
  if (!interactive && !a.has(L"prompt") && a.get(L"prompt-file").empty() &&
      prompt_w.empty() && _isatty(_fileno(stdin))) {
    interactive = true;   // 直接运行且没给提示词：当成聊天模式
    if (!cli_json() && !cli_quiet()) {
      std::cout << "（交互模式：输入内容回车发送，exit 退出）" << std::endl;
    }
  }
  const bool multi_turn = !a.has(L"no-context");
  const bool template_mode = a.has(L"template");
  const bool stream = !a.has(L"no-stream") && !a.has(L"json");

  // 提示词通道：auto（默认）/ chat（对话模板）/ text / tokens
  LlamaPromptMode prompt_mode = LlamaPromptMode::kAuto;
  {
    std::wstring cm = a.get(L"chat-mode");
    if (cm == L"chat" || cm == L"template" || cm == L"on") {
      prompt_mode = LlamaPromptMode::kChat;
    } else if (cm == L"text") {
      prompt_mode = LlamaPromptMode::kText;
    } else if (cm == L"tokens" || cm == L"token") {
      prompt_mode = LlamaPromptMode::kTokens;
    } else if (cm == L"auto" || cm.empty()) {
      prompt_mode = LlamaPromptMode::kAuto;
    } else {
      cli_error("--chat-mode 取值应为 auto|chat|text|tokens");
      return 2;
    }
  }

  LlamaSamplingParams sampling;
  sampling.max_tokens = (int)std::clamp<long long>(a.get_int(L"max-tokens", 128), 1, 8192);
  sampling.temperature = std::clamp(a.get_double(L"temperature", 0.8), 0.01, 4.0);
  sampling.top_k = (int)std::clamp<long long>(a.get_int(L"top-k", 40), 0, 100000);
  sampling.top_p = std::clamp(a.get_double(L"top-p", 0.95), 0.0, 1.0);
  sampling.repeat_penalty = std::clamp(a.get_double(L"repeat", 1.1), 0.5, 3.0);
  sampling.seed = (int)a.get_int(L"seed", -1);
  // 问答模板的停止词只对原始补全有效；对话模板由模型自己收尾
  if (template_mode && prompt_mode != LlamaPromptMode::kChat) {
    sampling.stop = {"\n问：", "\n答："};
  }

  const std::string system_prompt = cli_utf8(a.get(L"system"));

  // ---- 上下文文件 ----
  std::string initial_context;
  const std::wstring context_file = a.get(L"context");
  if (!context_file.empty()) {
    bool ok = false;
    initial_context = read_file_text(fs::path(context_file), &ok);
    if (!ok) {
      cli_error("无法读取上下文文件: " + cli_utf8(context_file));
      return 2;
    }
  }
  const std::wstring save_context_file = a.get(L"save-context");

  std::string exit_context;  // 用于写回

  // ---- 提示词 ----
  std::string prompt;
  if (!prompt_w.empty()) {
    if (prompt_w == L"-") {
      prompt = read_all_stdin();
    } else {
      prompt = cli_utf8(prompt_w);
    }
  } else if (!prompt_file.empty()) {
    bool ok = false;
    prompt = read_file_text(fs::path(prompt_file), &ok);
    if (!ok) {
      cli_error("无法读取提示词文件: " + cli_utf8(prompt_file));
      return 2;
    }
  } else if (!interactive) {
    prompt = read_all_stdin();
  }
  if (prompt.empty() && !interactive) {
    cli_error("没有输入提示词（用 --prompt / --prompt-file / 标准输入 / --interactive）");
    return 2;
  }
  while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r')) {
    prompt.pop_back();
  }

  // ---- 运行 ----
  int exit_code = 0;
  if (want_llama) {
    // 训练模型 → 导出 GGUF
    fs::path gguf = model_path;
    if (!is_gguf) {
      const fs::path out_dir = fs::path(Config::project_root()) / "models" / "gguf";
      gguf = out_dir / fs::path(model_path.filename().wstring() + L".gguf");
      std::error_code ec;
      bool need_export = !fs::exists(gguf, ec);
      if (!need_export) {
        need_export = fs::last_write_time(gguf, ec) <
                      fs::last_write_time(model_path / "model.pt", ec);
      }
      if (need_export) {
        cli_say("[准备] 导出 GGUF: " + u8(gguf));
        GgufExportResult result;
        std::string err;
        bool ok = false;
        try {
          ok = export_gpt_to_gguf(model_path, gguf, fs::path(), &result, &err);
        } catch (const std::exception& e) {
          err = e.what();
        }
        if (!ok) {
          cli_error("导出 GGUF 失败: " + err);
          return 1;
        }
      }
    }

    LlamaSession session;
    session.sampling = sampling;
    session.system_prompt = system_prompt;
    session.template_mode = template_mode;
    session.stream = stream;
    session.mode = prompt_mode;
    session.context = multi_turn ? initial_context : std::string();

    const fs::path server = server_w.empty()
                                ? LlamaEngine::detect_server_exe()
                                : fs::path(server_w);
    if (server.empty()) {
      cli_error("找不到 llama-server.exe，请用 --server 指定，或 --engine builtin");
      return 2;
    }
    session.opts.server_exe = server;
    session.opts.model_path = gguf;
    session.opts.n_ctx = (int)std::clamp<long long>(a.get_int(L"ctx", 2048), 64, 262144);
    session.opts.n_threads = (int)std::clamp<long long>(a.get_int(L"threads", 0), 0, 256);
    session.opts.n_gpu_layers = (int)std::clamp<long long>(a.get_int(L"ngl", 99), 0, 999);
    session.sp = load_spm_for_gguf(gguf);
    session.max_seq_len = 4096;

    std::string err;
    if (!session.engine.start(session.opts, &err)) {
      cli_error("llama-server 启动失败: " + err);
      return 1;
    }
    if (!cli_json()) {
      std::cerr << "[引擎] " << u8(server) << "\n[模型] " << u8(gguf) << std::endl;
    }

    if (!interactive) {
      std::string answer;
      const bool ok = session.ask(prompt, &answer, &err);
      if (!ok) {
        cli_error("生成失败: " + err);
        session.engine.stop();
        return 1;
      }
      if (stream) std::cout << std::endl;
      exit_context = session.context;
      if (cli_json()) {
        std::cout << "{\"ok\":true,\"engine\":\"llama\",\"model\":\""
                  << cli_json_escape(u8(gguf)) << "\",\"text\":\""
                  << cli_json_escape(answer) << "\",\"chars\":" << answer.size()
                  << "}" << std::endl;
      }
    } else {
      std::string line;
      while (true) {
        std::cout << "\n你> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        while (!line.empty() && (line.back() == '\r')) line.pop_back();
        if (line == "exit" || line == "quit" || line == ":q") break;
        if (line.empty()) continue;
        std::cout << "模型> " << std::flush;
        std::string answer;
        if (!session.ask(line, &answer, &err)) {
          cli_error("生成失败: " + err);
          exit_code = 1;
          break;
        }
        std::cout << std::endl;
      }
      exit_context = session.context;
    }
    session.engine.stop();
  } else {
    // ---- 内置引擎 ----
    if (is_gguf) {
      cli_error("内置 libtorch 引擎只能加载 saves/modelN（GGUF 请用 llama.cpp 引擎）");
      return 2;
    }
    const fs::path config_path = model_path / "config.json";
    if (!fs::exists(config_path) || !fs::exists(model_path / "model.pt")) {
      cli_error("模型目录不完整（需要 config.json + model.pt）: " + u8(model_path));
      return 2;
    }
    GPTConfig mcfg;
    try {
      std::ifstream f(config_path, std::ios::binary);
      std::ostringstream ss;
      ss << f.rdbuf();
      const Json js = Json::parse(ss.str());
      mcfg.vocab_size = js.get_int("vocab_size").value_or(8000);
      mcfg.embed_dim = js.get_int("embed_dim").value_or(256);
      mcfg.num_heads = js.get_int("num_heads").value_or(8);
      mcfg.num_layers = js.get_int("num_layers").value_or(6);
      mcfg.max_seq_len = js.get_int("max_seq_len").value_or(256);
    } catch (const std::exception& e) {
      cli_error(std::string("解析 config.json 失败: ") + e.what());
      return 1;
    }

    const fs::path spm_path =
        find_matching_spm(model_path, mcfg.vocab_size, fs::path(spm_w));
    if (spm_path.empty()) {
      cli_error("找不到与模型词表大小 (" + std::to_string(mcfg.vocab_size) +
                ") 匹配的 spm.model，请用 --spm 指定");
      return 2;
    }

    BuiltinSession session;
    session.template_mode = template_mode;
    session.temperature = sampling.temperature;
    session.top_k = sampling.top_k;
    session.max_tokens = sampling.max_tokens;
    session.max_seq_len = mcfg.max_seq_len;
    session.context = multi_turn ? initial_context : std::string();
    try {
      const torch::Device dev = torch::cuda::is_available()
                                    ? torch::Device(torch::kCUDA)
                                    : torch::Device(torch::kCPU);
      session.model = load_gpt_from_saves(model_path, dev);
      session.sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
      if (!session.sp->Load(path_to_ansi(spm_path)).ok()) {
        cli_error("分词器加载失败: " + u8(spm_path));
        return 1;
      }
    } catch (const std::exception& e) {
      cli_error(std::string("加载模型失败: ") + e.what());
      return 1;
    }
    if (!cli_json()) {
      std::cerr << "[引擎] builtin (libtorch)\n[模型] " << u8(model_path)
                << "\n[分词器] " << u8(spm_path) << std::endl;
    }

    std::string err;
    if (!interactive) {
      std::string answer;
      if (!session.ask(prompt, &answer, &err)) {
        cli_error("生成失败: " + err);
        return 1;
      }
      std::cout << answer << std::endl;
      exit_context = session.context;
      if (cli_json()) {
        std::cout << "{\"ok\":true,\"engine\":\"builtin\",\"model\":\""
                  << cli_json_escape(u8(model_path)) << "\",\"text\":\""
                  << cli_json_escape(answer) << "\",\"chars\":" << answer.size()
                  << "}" << std::endl;
      }
    } else {
      std::string line;
      while (true) {
        std::cout << "\n你> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "exit" || line == "quit" || line == ":q") break;
        if (line.empty()) continue;
        std::cout << "模型> " << std::flush;
        std::string answer;
        if (!session.ask(line, &answer, &err)) {
          cli_error("生成失败: " + err);
          exit_code = 1;
          break;
        }
        std::cout << answer << std::endl;
      }
      exit_context = session.context;
    }
  }

  if (!save_context_file.empty()) {
    if (!write_file_text(fs::path(save_context_file), exit_context)) {
      cli_error("无法写入上下文文件: " + cli_utf8(save_context_file));
      if (exit_code == 0) exit_code = 1;
    } else {
      cli_say("[上下文已保存] " + cli_utf8(save_context_file));
    }
  }
  return exit_code;
}

} // namespace gpt
