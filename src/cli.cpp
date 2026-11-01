#include "cli.h"

#include <torch/torch.h>

#include <sentencepiece_processor.h>

#include <algorithm>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <ctime>
#include <thread>
#include <io.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "config.h"
#include "dataset.h"
#include "gguf_writer.h"
#include "json_util.h"
#include "legacy_serial.h"
#include "llama_engine.h"
#include "model.h"
#include "path_util.h"
#include "tokenizer_worker.h"
#include "train.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

bool g_quiet = false;
bool g_json = false;

std::wstring lower(std::wstring s) {
  for (auto& c : s) c = (wchar_t)towlower(c);
  return s;
}

// 短选项 → 长选项
std::wstring expand_short(const std::wstring& raw) {
  static const std::map<std::wstring, std::wstring> alias = {
      {L"h", L"help"},      {L"q", L"quiet"},   {L"j", L"json"},
      {L"v", L"verbose"},   {L"m", L"model"},   {L"o", L"out"},
      {L"p", L"prompt"},    {L"n", L"max-tokens"}, {L"t", L"threads"},
      {L"c", L"ctx"},       {L"d", L"data"},    {L"s", L"steps"},
      {L"e", L"epochs"},    {L"b", L"batch"},   {L"r", L"resume"},
  };
  if (raw.size() >= 2 && raw[0] == L'-' && raw[1] != L'-') {
    const std::wstring key = lower(raw.substr(1));
    const auto it = alias.find(key);
    if (it != alias.end()) return it->second;
    return key;
  }
  return {};
}

bool is_flag_key(const std::wstring& key) {
  static const std::set<std::wstring> flags = {
      L"help",        L"quiet",      L"json",       L"verbose",
      L"resume",      L"cpu",        L"template",   L"interactive",
      L"repl",        L"no-context", L"no-stream",  L"print-only",
      L"probe-server", L"no-token",  L"version",    L"gui",
  };
  return flags.count(key) > 0;
}

std::wstring bdata(const std::vector<std::wstring>& v) {
  return v.empty() ? std::wstring() : v.front();
}

} // namespace

// ===========================================================================
// 参数解析
// ===========================================================================
bool CliArgs::has(const std::wstring& key) const {
  return opts.find(lower(key)) != opts.end();
}

std::wstring CliArgs::get(const std::wstring& key,
                          const std::wstring& def) const {
  const auto it = opts.find(lower(key));
  if (it == opts.end() || it->second.empty()) return def;
  return it->second.front();
}

long long CliArgs::get_int(const std::wstring& key, long long def) const {
  const std::wstring v = get(key);
  if (v.empty()) return def;
  wchar_t* end = nullptr;
  const long long r = wcstoll(v.c_str(), &end, 10);
  return (end != nullptr && *end == 0) ? r : def;
}

double CliArgs::get_double(const std::wstring& key, double def) const {
  const std::wstring v = get(key);
  if (v.empty()) return def;
  wchar_t* end = nullptr;
  const double r = wcstod(v.c_str(), &end);
  return (end != nullptr && *end == 0) ? r : def;
}

std::vector<std::wstring> CliArgs::get_list(const std::wstring& key) const {
  const auto it = opts.find(lower(key));
  return it == opts.end() ? std::vector<std::wstring>() : it->second;
}

std::wstring CliArgs::pos(size_t index, const std::wstring& def) const {
  return index < positional.size() ? positional[index] : def;
}

CliArgs parse_cli(const std::vector<std::wstring>& argv, size_t start) {
  CliArgs out;
  for (size_t i = start; i < argv.size(); ++i) {
    const std::wstring& tok = argv[i];
    if (tok.size() < 2 || tok[0] != L'-') {
      out.positional.push_back(tok);
      continue;
    }
    // 去掉前导 '-'，再拆 --key=value
    std::wstring body = tok;
    while (!body.empty() && body[0] == L'-') body.erase(body.begin());
    std::wstring inline_value;
    const size_t eq = body.find(L'=');
    if (eq != std::wstring::npos) {
      inline_value = body.substr(eq + 1);
      body = body.substr(0, eq);
    }
    // 短选项展开（-m → model），长选项直接用名字
    std::wstring key;
    if (tok.size() >= 2 && tok[1] != L'-') {
      key = expand_short(std::wstring(L"-") + body);
      if (key.empty()) key = lower(body);
    } else {
      key = lower(body);
    }
    if (!inline_value.empty()) {
      out.opts[key].push_back(inline_value);
      continue;
    }
    // 下一个 token 作为值：不是选项（"-" 开头的负数仍算值）
    if (i + 1 < argv.size()) {
      const std::wstring& nx = argv[i + 1];
      const bool is_negative_number =
          nx.size() >= 2 && nx[0] == L'-' && iswdigit(nx[1]);
      const bool is_option = nx.size() >= 2 && nx[0] == L'-' && !is_negative_number;
      if (!is_option) {
        out.opts[key].push_back(nx);
        ++i;
        continue;
      }
    }
    out.opts[key].push_back(L"1");
  }
  return out;
}

// ===========================================================================
// 输出
// ===========================================================================
void cli_set_quiet(bool quiet) { g_quiet = quiet; }
void cli_set_json(bool json) { g_json = json; }
bool cli_quiet() { return g_quiet; }
bool cli_json() { return g_json; }

std::string cli_utf8(const std::wstring& w) { return wide_to_utf8(w); }

std::string cli_time() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_s(&tm, &now);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", tm.tm_hour, tm.tm_min,
                tm.tm_sec);
  return buf;
}

void cli_say(const std::string& line) {
  if (g_quiet) return;
  std::cout << line << std::endl;
}

void cli_error(const std::string& line) {
  std::cerr << line << std::endl;
}

std::string cli_json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += (char)c;
        }
    }
  }
  return out;
}

std::wstring cli_default_raw_dir() {
  return (fs::path(Config::project_root()) / "data" / "data" / "raw").wstring();
}

namespace {

// 默认数据集（配置路径 → 项目内常见位置）
fs::path find_default_data() {
  fs::path p = Config::defaults().data_path;
  if (fs::exists(p)) return p;
  const fs::path root = Config::project_root();
  const fs::path candidates[] = {
      root / "data" / "data" / "processed" / "token_ids.txt",
      root / "data" / "processed" / "token_ids.txt",
      root / "data" / "token_ids.txt"};
  for (const auto& c : candidates) {
    std::error_code ec;
    if (fs::is_regular_file(c, ec)) return c;
  }
  return {};
}

void print_train_help() {
  std::cout <<
      "用法: gpt_trainer.exe train [选项]\n"
      "\n"
      "数据与结构：\n"
      "  --data <token_ids.txt>   训练数据（默认自动查找 data/**/token_ids.txt）\n"
      "  --seq-len <n>            序列长度（默认 256）\n"
      "  --stride <n>             滑动窗口步长（默认 128）\n"
      "  --embed <n>              嵌入维度（默认 256，必须能被 --heads 整除）\n"
      "  --heads <n>              注意力头数（默认 8）\n"
      "  --layers <n>             层数（默认 6）\n"
      "  --dropout <f>            丢弃率（默认 0.1）\n"
      "  --vocab <n>              词表大小（默认读 config 的 8000）\n"
      "\n"
      "优化与轮次：\n"
      "  --lr <f>                 学习率（默认 3e-4）\n"
      "  --batch <n>              批次大小（默认 16）\n"
      "  --epochs <n>             训练轮数（默认 10，--steps 优先）\n"
      "  --steps <n>              总训练步数（给定后忽略 epochs）\n"
      "  --weight-decay <f>       权重衰减（默认 0.01）\n"
      "  --grad-clip <f>          梯度裁剪（默认 1.0）\n"
      "  --ckpt-interval <n>      断点保存间隔步数（默认 5000，0=关闭）\n"
      "\n"
      "断点 / 设备：\n"
      "  --resume                 从 models/training_state 继续训练\n"
      "  --init <model.pt>        用已有权重初始化（优化器重新开始）\n"
      "  --cpu                    强制使用 CPU\n"
      "  --threads <n>            训练时 torch 线程数（默认自动）\n"
      "\n"
      "示例：\n"
      "  gpt_trainer.exe train --data data/data/processed/token_ids.txt --epochs 3 --batch 8\n"
      "  gpt_trainer.exe train --steps 2000 --lr 0.0005 --embed 512 --heads 8 --layers 8\n";
}

void print_tokenize_help() {
  std::cout <<
      "用法: gpt_trainer.exe tokenize [选项] [语料.txt ...]\n"
      "\n"
      "  --out <目录>             输出目录（默认 <项目根>/data）\n"
      "  --corpus <文件/目录>      语料，可重复；也可直接给位置参数\n"
      "  --threads <n>            训练线程数（0=自动）\n"
      "  --vocab <n>              词表大小（0=自动）\n"
      "  --sentences <n>          最大训练句子数（0=自动/全部）\n"
      "\n"
      "产物：<out>/models/spm.model、<out>/data/processed/token_ids.txt、\n"
      "      <out>/data/config.json、<out>/dataset.zip\n"
      "\n"
      "示例：\n"
      "  gpt_trainer.exe tokenize --out D:\\data --vocab 8000 corpus1.txt corpus2.txt\n"
      "  gpt_trainer.exe tokenize --out D:\\data            # 使用 data/raw 目录\n";
}

void print_export_help() {
  std::cout <<
      "用法: gpt_trainer.exe export-gguf [选项] [模型目录] [输出.gguf]\n"
      "\n"
      "  --model <saves/modelN>   模型目录（含 config.json + model.pt）\n"
      "  --out <文件.gguf>        输出路径（默认 models/gguf/<模型名>.gguf）\n"
      "  --spm <spm.model>        指定分词器（默认自动挑选词表匹配的）\n"
      "\n"
      "示例：\n"
      "  gpt_trainer.exe export-gguf --model saves/model1\n"
      "  gpt_trainer.exe export-gguf saves/model1 D:\\m.gguf --spm models/spm.model\n";
}

void print_list_help() {
  std::cout <<
      "用法: gpt_trainer.exe list [--json]\n"
      "\n"
      "列出 saves 中的训练模型与 models/gguf 中的 GGUF（含大小）。\n";
}

void print_info_help() {
  std::cout <<
      "用法: gpt_trainer.exe info [--probe-server] [--json]\n"
      "\n"
      "  --probe-server           额外运行 llama-server --version 探测版本\n";
}

void print_serve_help() {
  std::cout <<
      "用法: gpt_trainer.exe serve --model <模型.gguf> [选项]\n"
      "\n"
      "  --server <llama-server.exe>  引擎路径（默认自动探测）\n"
      "  --ctx <n>                    上下文长度（默认 2048）\n"
      "  --threads <n>                CPU 线程数（默认 0=自动）\n"
      "  --ngl <n>                    GPU 层数（默认 99，0=纯 CPU）\n"
      "  --port <n>                   端口（默认自动）\n"
      "  --print-only                 只打印 llama-server 命令行，不启动\n"
      "\n"
      "启动后按 Ctrl+C 退出（llama-server 会随主进程一起结束）。\n";
}



std::string human_size(uintmax_t bytes) {
  std::ostringstream oss;
  const double mb = (double)bytes / (1024.0 * 1024.0);
  if (mb >= 1024.0) {
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << (mb / 1024.0) << " GB";
  } else {
    oss.setf(std::ios::fixed);
    oss.precision(1);
    oss << mb << " MB";
  }
  return oss.str();
}

} // namespace

// ===========================================================================
// train
// ===========================================================================
int cli_cmd_train(const CliArgs& a, bool legacy) {
  (void)legacy;
  if (a.has(L"help")) {
    print_train_help();
    return 0;
  }
  Config cfg = Config::defaults();

  fs::path data = a.has(L"data") ? fs::path(a.get(L"data")) : find_default_data();
  if (data.empty() || !fs::exists(data)) {
    cli_error("找不到训练数据，请用 --data 指定 token_ids.txt（可以用「tokenize」命令生成）");
    cli_error("当前默认路径: " + path_to_utf8(cfg.data_path));
    return 2;
  }
  cfg.data_path = data;

  cfg.learning_rate = a.get_double(L"lr", cfg.learning_rate);
  cfg.batch_size = a.get_int(L"batch", cfg.batch_size);
  cfg.max_seq_len = a.get_int(L"seq-len", cfg.max_seq_len);
  cfg.num_epochs = a.get_int(L"epochs", cfg.num_epochs);
  cfg.stride = a.get_int(L"stride", cfg.stride);
  cfg.embed_dim = a.get_int(L"embed", cfg.embed_dim);
  cfg.num_heads = a.get_int(L"heads", cfg.num_heads);
  cfg.num_layers = a.get_int(L"layers", cfg.num_layers);
  cfg.dropout = a.get_double(L"dropout", cfg.dropout);
  cfg.weight_decay = a.get_double(L"weight-decay", cfg.weight_decay);
  cfg.grad_clip = a.get_double(L"grad-clip", cfg.grad_clip);
  cfg.vocab_size = a.get_int(L"vocab", cfg.vocab_size);
  // 未显式指定 --vocab 时，从数据目录的 config.json 读取（tokenize 生成，避免词表不匹配）
  if (!a.has(L"vocab")) {
    for (const fs::path& dir : {data.parent_path(), data.parent_path().parent_path()}) {
      const fs::path cfgp = dir / "config.json";
      std::error_code ec;
      if (!fs::is_regular_file(cfgp, ec)) continue;
      std::ifstream f(cfgp, std::ios::binary);
      if (!f) continue;
      std::ostringstream ss;
      ss << f.rdbuf();
      try {
        const Json js = Json::parse(ss.str());
        const auto v = js.get_int("vocab_size");
        if (v.has_value() && *v > 0) {
          cfg.vocab_size = *v;
          cli_say("词表大小（来自 " + path_to_utf8(cfgp) + "）: " +
                  std::to_string(cfg.vocab_size));
          break;
        }
      } catch (...) {
      }
    }
  }
  cfg.checkpoint_interval = a.get_int(L"ckpt-interval", cfg.checkpoint_interval);
  if (a.has(L"steps")) cfg.total_steps = a.get_int(L"steps", 0);
  if (a.has(L"cpu")) cfg.use_cuda = false;
  if (a.has(L"threads")) {
    const int threads = (int)a.get_int(L"threads", 0);
    if (threads > 0) torch::set_num_threads(threads);
  }

  if (cfg.embed_dim % cfg.num_heads != 0) {
    cli_error("嵌入维度 (" + std::to_string(cfg.embed_dim) +
              ") 必须能被注意力头数 (" + std::to_string(cfg.num_heads) + ") 整除");
    return 2;
  }
  if (cfg.batch_size <= 0 || cfg.max_seq_len <= 1 || cfg.stride <= 0) {
    cli_error("参数非法：--batch / --seq-len / --stride 必须为正");
    return 2;
  }

  cfg.init_dirs();

  TrainerOptions opts;
  opts.resume = a.has(L"resume");
  if (a.has(L"init")) {
    opts.init_model = fs::path(a.get(L"init"));
    if (!fs::exists(opts.init_model)) {
      cli_error("初始权重不存在: " + path_to_utf8(opts.init_model));
      return 2;
    }
  }

  if (!g_json && !g_quiet) {
    std::cout << "=== 训练 ===\n"
              << "数据: " << path_to_utf8(cfg.data_path) << "\n"
              << "设备: " << (cfg.use_cuda ? "CUDA" : "CPU") << "\n"
              << "结构: embed=" << cfg.embed_dim << " heads=" << cfg.num_heads
              << " layers=" << cfg.num_layers << " seq=" << cfg.max_seq_len
              << " vocab=" << cfg.vocab_size << "\n"
              << "优化: lr=" << cfg.learning_rate << " batch=" << cfg.batch_size
              << " stride=" << cfg.stride << " wd=" << cfg.weight_decay << "\n"
              << "轮次: "
              << (cfg.total_steps.has_value()
                      ? ("steps=" + std::to_string(*cfg.total_steps))
                      : ("epochs=" + std::to_string(cfg.num_epochs)))
              << " ckpt_interval=" << cfg.checkpoint_interval
              << (opts.resume ? " [断点续训]" : "")
              << (opts.init_model.empty()
                      ? ""
                      : (" [init=" + path_to_utf8(opts.init_model) + "]"))
              << std::endl;
  }

  float last_loss = 0.0f;
  TrainerCallbacks cb;
  cb.on_log = [](const std::string& line) { cli_say(line); };
  cb.on_progress = [&last_loss](float loss, int epoch, long long step,
                                long long total) {
    last_loss = loss;
    if (g_quiet || g_json) return;
    std::cout << "  [epoch " << epoch << "] step " << step << "/" << total
              << "  loss=" << loss << std::endl;
  };

  int exit_code = 0;
  try {
    Trainer trainer(cfg, cb, opts);
    trainer.train();
    const double final_loss =
        last_loss != 0.0f ? (double)last_loss : (double)trainer.current_loss;
    if (g_json) {
      std::cout << "{\"ok\":true,\"steps\":" << trainer.step
                << ",\"loss\":" << final_loss << ",\"saves\":\""
                << cli_json_escape(path_to_utf8(cfg.save_root)) << "\"}"
                << std::endl;
    } else {
      std::cout << "训练完成: steps=" << trainer.step << " loss=" << final_loss
                << "\n模型目录: " << path_to_utf8(cfg.save_root) << std::endl;
    }
  } catch (const std::exception& e) {
    cli_error(std::string("训练失败: ") + e.what());
    exit_code = 1;
  }
  return exit_code;
}
// ===========================================================================
// tokenize
// ===========================================================================
int cli_cmd_tokenize(const CliArgs& a, bool legacy) {
  if (a.has(L"help")) {
    print_tokenize_help();
    return 0;
  }
  TokenizerTask task;
  std::vector<std::wstring> corpora;

  size_t first_pos = 0;
  if (legacy && !a.has(L"out")) {
    // 旧式：--tokenize <输出目录> <语料...>
    if (!a.positional.empty()) {
      task.output_dir = fs::path(a.pos(0));
      first_pos = 1;
    }
  } else if (a.has(L"out")) {
    task.output_dir = fs::path(a.get(L"out"));
  }
  if (task.output_dir.empty()) {
    task.output_dir = fs::path(Config::project_root()) / "data";
  }
  for (size_t i = first_pos; i < a.positional.size(); ++i) {
    corpora.push_back(a.positional[i]);
  }
  for (const auto& c : a.get_list(L"corpus")) corpora.push_back(c);

  for (const auto& c : corpora) {
    const fs::path p(c);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
      for (const auto& entry : fs::directory_iterator(p, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".txt") {
          task.input_files.push_back(entry.path());
        }
      }
    } else if (fs::is_regular_file(p, ec)) {
      task.input_files.push_back(p);
    } else {
      cli_error("忽略不存在的语料: " + path_to_utf8(p));
    }
  }

  task.num_threads = (int)a.get_int(L"threads", 0);
  task.vocab_size = (int)a.get_int(L"vocab", 0);
  task.input_sentence_size = (int)a.get_int(L"sentences", 0);

  std::error_code ec;
  fs::create_directories(task.output_dir, ec);
  if (!g_quiet) {
    std::cout << "=== 分词器训练 ===\n"
              << "输出目录: " << path_to_utf8(task.output_dir) << "\n"
              << "语料: "
              << (task.input_files.empty()
                      ? ("默认 data/raw 目录")
                      : (std::to_string(task.input_files.size()) + " 个文件"))
              << std::endl;
  }

  task.log = [](const std::string& line) { cli_say("  " + line); };

  TokenizerResult result;
  std::string err;
  bool ok = false;
  try {
    ok = run_tokenizer_task(task, &result, &err);
  } catch (const std::exception& e) {
    err = e.what();
  }
  if (!ok) {
    cli_error("分词器训练失败: " + err);
    return 1;
  }
  if (g_json) {
    std::cout << "{\"ok\":true"
              << ",\"spm\":\"" << cli_json_escape(path_to_utf8(result.spm_model)) << "\""
              << ",\"token_ids\":\"" << cli_json_escape(path_to_utf8(result.token_ids)) << "\""
              << ",\"config\":\"" << cli_json_escape(path_to_utf8(result.config_json)) << "\""
              << ",\"dataset\":\"" << cli_json_escape(path_to_utf8(result.dataset_zip)) << "\""
              << ",\"vocab_size\":" << result.vocab_size
              << ",\"threads\":" << result.num_threads
              << ",\"sentences\":" << result.input_sentence_size << "}" << std::endl;
  } else {
    std::cout << "分词器训练完成:\n"
              << "  spm.model      : " << path_to_utf8(result.spm_model) << "\n"
              << "  token_ids.txt  : " << path_to_utf8(result.token_ids) << "\n"
              << "  config.json    : " << path_to_utf8(result.config_json) << "\n"
              << "  dataset.zip    : " << path_to_utf8(result.dataset_zip) << "\n"
              << "  词表大小       : " << result.vocab_size << "\n"
              << "训练数据可直接用于: gpt_trainer.exe train --data \""
              << path_to_utf8(result.token_ids) << "\"" << std::endl;
  }
  return 0;
}

// ===========================================================================
// export-gguf
// ===========================================================================
int cli_cmd_export_gguf(const CliArgs& a, bool legacy) {
  if (a.has(L"help")) {
    print_export_help();
    return 0;
  }
  fs::path model_dir = a.has(L"model") ? fs::path(a.get(L"model")) : fs::path();
  fs::path out = a.has(L"out") ? fs::path(a.get(L"out")) : fs::path();
  fs::path spm = a.has(L"spm") ? fs::path(a.get(L"spm")) : fs::path();
  if (legacy) {
    if (model_dir.empty() && a.positional.size() >= 1) model_dir = fs::path(a.pos(0));
    if (out.empty() && a.positional.size() >= 2) out = fs::path(a.pos(1));
    if (spm.empty() && a.positional.size() >= 3) spm = fs::path(a.pos(2));
  } else {
    if (model_dir.empty() && a.positional.size() >= 1) model_dir = fs::path(a.pos(0));
    if (out.empty() && a.positional.size() >= 2) out = fs::path(a.pos(1));
  }
  if (model_dir.empty()) {
    cli_error("请用 --model 指定模型目录（saves/modelN）");
    return 2;
  }
  if (out.empty()) {
    out = fs::path(Config::project_root()) / "models" / "gguf" /
          fs::path(model_dir.filename().wstring() + L".gguf");
  }

  GgufExportResult result;
  std::string err;
  bool ok = false;
  try {
    ok = export_gpt_to_gguf(model_dir, out, spm, &result, &err);
  } catch (const std::exception& e) {
    err = e.what();
  }
  if (!ok) {
    cli_error("导出失败: " + err);
    return 1;
  }
  if (g_json) {
    std::cout << "{\"ok\":true"
              << ",\"gguf\":\"" << cli_json_escape(path_to_utf8(result.out_path)) << "\""
              << ",\"spm\":\"" << cli_json_escape(path_to_utf8(result.spm_model)) << "\""
              << ",\"tensors\":" << result.tensor_count
              << ",\"vocab_size\":" << result.vocab_size
              << ",\"bytes\":" << result.bytes << "}" << std::endl;
  } else if (!g_quiet) {
    std::cout << "GGUF 导出成功:\n"
              << "  文件    : " << path_to_utf8(result.out_path) << "\n"
              << "  分词器  : " << path_to_utf8(result.spm_model) << "\n"
              << "  张量数  : " << result.tensor_count << "\n"
              << "  词表大小: " << result.vocab_size << "\n"
              << "  体积    : " << human_size((uintmax_t)result.bytes) << std::endl;
  } else {
    std::cout << path_to_utf8(result.out_path) << std::endl;
  }
  if (!err.empty()) cli_say("提示: " + err);
  return 0;
}

// ===========================================================================
// list
// ===========================================================================
int cli_cmd_list(const CliArgs& a) {
  if (a.has(L"help")) {
    print_list_help();
    return 0;
  }
  const Config cfg = Config::defaults();
  struct Entry {
    std::string name;
    std::string kind;
    std::string path;
    uintmax_t bytes = 0;
    std::string mtime;
  };
  std::vector<Entry> entries;
  std::error_code ec;

  // saves 模型
  if (fs::exists(cfg.save_root, ec)) {
    for (const auto& entry : fs::directory_iterator(cfg.save_root, ec)) {
      if (!entry.is_directory()) continue;
      Entry e;
      e.name = path_to_utf8(entry.path().filename());
      e.kind = "saves";
      e.path = path_to_utf8(entry.path());
      const fs::path model_pt = entry.path() / "model.pt";
      if (fs::is_regular_file(model_pt, ec)) e.bytes = fs::file_size(model_pt, ec);
      entries.push_back(e);
    }
  }
  // models/gguf
  const fs::path gguf_dir = fs::path(cfg.project_root()) / "models" / "gguf";
  if (fs::exists(gguf_dir, ec)) {
    for (const auto& entry : fs::directory_iterator(gguf_dir, ec)) {
      if (!entry.is_regular_file(ec) || entry.path().extension() != ".gguf") continue;
      Entry e;
      e.name = path_to_utf8(entry.path().filename());
      e.kind = "gguf";
      e.path = path_to_utf8(entry.path());
      e.bytes = fs::file_size(entry.path(), ec);
      entries.push_back(e);
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry& x, const Entry& y) {
              if (x.kind != y.kind) return x.kind < y.kind;
              return x.name < y.name;
            });

  if (g_json) {
    std::cout << "{\"count\":" << entries.size() << ",\"items\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << "{\"name\":\"" << cli_json_escape(entries[i].name) << "\",\"kind\":\""
                << entries[i].kind << "\",\"path\":\"" << cli_json_escape(entries[i].path)
                << "\",\"bytes\":" << entries[i].bytes << "}";
    }
    std::cout << "]}" << std::endl;
    return 0;
  }
  if (entries.empty()) {
    std::cout << "（暂无模型：先 train 训练，或 export-gguf 导出）" << std::endl;
    return 0;
  }
  std::cout << "类型    名称                          体积        路径\n";
  std::cout << "--------------------------------------------------------------\n";
  for (const auto& e : entries) {
    std::printf("%-7s %-28s %-11s %s\n", e.kind.c_str(), e.name.c_str(),
                human_size(e.bytes).c_str(), e.path.c_str());
  }
  return 0;
}

// ===========================================================================
// info
// ===========================================================================
int cli_cmd_info(const CliArgs& a) {
  if (a.has(L"help")) {
    print_info_help();
    return 0;
  }
  const Config cfg = Config::defaults();
  const bool cuda = cfg.use_cuda;
  int cuda_count = 0;
  try {
    cuda_count = (int)torch::cuda::device_count();
  } catch (...) {
    cuda_count = 0;
  }
  const fs::path root = cfg.project_root();
  const fs::path llama_server = LlamaEngine::detect_server_exe();
  fs::path default_data = find_default_data();
  auto spm_candidates = find_spm_candidates(fs::path());

  std::string llama_version;
  if (a.has(L"probe-server") && !llama_server.empty()) {
    const std::wstring cmd = L"\"" + llama_server.wstring() + L"\" --version";
    FILE* pipe = _wpopen(cmd.c_str(), L"rt");
    if (pipe != nullptr) {
      char buf[512];
      while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        llama_version += buf;
        if (llama_version.size() > 2048) break;
      }
      _pclose(pipe);
    }
  }

  if (g_json) {
    std::cout << "{";
    std::cout << "\"project_root\":\"" << cli_json_escape(path_to_utf8(root)) << "\"";
    std::cout << ",\"cuda\":" << (cuda ? "true" : "false");
    std::cout << ",\"cuda_devices\":" << cuda_count;
    std::cout << ",\"data\":\"" << cli_json_escape(path_to_utf8(default_data)) << "\"";
    std::cout << ",\"saves\":\"" << cli_json_escape(path_to_utf8(cfg.save_root)) << "\"";
    std::cout << ",\"models\":\"" << cli_json_escape(path_to_utf8(cfg.model_save_dir)) << "\"";
    std::cout << ",\"llama_server\":\"" << cli_json_escape(path_to_utf8(llama_server)) << "\"";
    std::cout << ",\"spm_candidates\":[";
    for (size_t i = 0; i < spm_candidates.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << "\"" << cli_json_escape(path_to_utf8(spm_candidates[i])) << "\"";
    }
    std::cout << "]";
    if (!llama_version.empty()) {
      std::cout << ",\"llama_version\":\"" << cli_json_escape(llama_version) << "\"";
    }
    std::cout << "}" << std::endl;
    return 0;
  }

  std::cout << "=== 运行环境 ===\n"
            << "项目根目录 : " << path_to_utf8(root) << "\n"
            << "设备       : " << (cuda ? "CUDA" : "CPU")
            << "（可选设备数 " << cuda_count << "）\n"
            << "默认数据集 : " << (default_data.empty() ? "（未找到）" : path_to_utf8(default_data))
            << "\n"
            << "saves      : " << path_to_utf8(cfg.save_root) << "\n"
            << "models     : " << path_to_utf8(cfg.model_save_dir) << "\n"
            << "logs       : " << path_to_utf8(cfg.log_dir) << "\n"
            << "llama.cpp  : "
            << (llama_server.empty() ? "（未找到 llama-server.exe）" : path_to_utf8(llama_server))
            << std::endl;
  if (!spm_candidates.empty()) {
    std::cout << "分词器候选 :\n";
    for (const auto& c : spm_candidates) {
      std::cout << "  - " << path_to_utf8(c) << "\n";
    }
  }
  if (!llama_version.empty()) {
    std::cout << "llama 版本 :\n" << llama_version;
  }
  return 0;
}

// ===========================================================================
// serve
// ===========================================================================
int cli_cmd_serve(const CliArgs& a) {
  if (a.has(L"help")) {
    print_serve_help();
    return 0;
  }
  const std::wstring model = a.get(L"model", a.pos(0));
  if (model.empty()) {
    cli_error("请用 --model 指定 GGUF 模型");
    return 2;
  }
  fs::path server = a.has(L"server") ? fs::path(a.get(L"server"))
                                     : LlamaEngine::detect_server_exe();
  if (server.empty()) {
    cli_error("找不到 llama-server.exe，请用 --server 指定");
    return 2;
  }
  const int ctx = (int)a.get_int(L"ctx", 2048);
  const int threads = (int)a.get_int(L"threads", 0);
  const int ngl = (int)a.get_int(L"ngl", 99);
  const int port = (int)a.get_int(L"port", 0);

  if (a.has(L"print-only")) {
    std::cout << "\"" << path_to_utf8(server) << "\" -m \"" << cli_utf8(model)
              << "\" --host 127.0.0.1 --port " << (port > 0 ? port : 8080)
              << " -c " << ctx;
    if (threads > 0) std::cout << " -t " << threads;
    std::cout << " -ngl " << ngl << std::endl;
    return 0;
  }

  LlamaEngineOptions opts;
  opts.server_exe = server;
  opts.model_path = fs::path(model);
  opts.n_ctx = ctx;
  opts.n_threads = threads;
  opts.n_gpu_layers = ngl;
  opts.port = port;

  LlamaEngine engine;
  std::string err;
  if (!engine.start(opts, &err)) {
    cli_error("llama-server 启动失败: " + err);
    return 1;
  }
  std::cout << "llama-server 已就绪（Ctrl+C 退出）\n"
            << engine.model_info() << "\n"
            << "日志: " << path_to_utf8(engine.log_path()) << std::endl;
  // 保持进程存活；llama-server 在 job object 中，主进程退出即结束
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!engine.running() || !engine.healthy()) {
      cli_error("llama-server 已退出");
      return 1;
    }
  }
}

// ===========================================================================
// selftest / loadtest（保留原有调试能力）
// ===========================================================================
int cli_cmd_selftest(const CliArgs& a) {
  if (a.has(L"help")) {
    cli_print_selftest_help();
    return 0;
  }
  const Config cfg = Config::defaults();
  std::cout << "=== GPT 互读自检 ===" << std::endl;
  const torch::Device dev = torch::cuda::is_available()
                                ? torch::Device(torch::kCUDA)
                                : torch::Device(torch::kCPU);
  std::cout << "设备: " << (dev.is_cuda() ? "cuda" : "cpu") << std::endl;

  const fs::path model_dir = a.has(L"model")
                                 ? fs::path(a.get(L"model"))
                                 : (cfg.project_root().parent_path() /
                                    fs::path(L"py训练器") / fs::path(L"saves") /
                                    fs::path(L"model1"));
  if (!fs::exists(model_dir / "model.pt")) {
    cli_error("未找到模型: " + path_to_utf8(model_dir));
    return 2;
  }
  auto model = load_gpt_from_saves(model_dir, dev);
  std::cout << "[OK] 已加载模型: " << path_to_utf8(model_dir) << std::endl;
  {
    torch::NoGradGuard guard;
    auto x = torch::randint(0, model->config.vocab_size, {2, 32},
                           torch::TensorOptions().dtype(torch::kInt64))
                 .to(dev);
    auto logits = model->forward(x);
    std::cout << "[OK] 前向输出形状: " << logits.sizes()
              << ", 均值: " << logits.mean().item<float>() << std::endl;
  }
  const fs::path spm_path =
      a.has(L"spm") ? fs::path(a.get(L"spm"))
                    : (cfg.project_root().parent_path() / fs::path(L"py训练器") /
                       fs::path(L"models") / fs::path(L"spm.model"));
  sentencepiece::SentencePieceProcessor sp;
  if (!fs::exists(spm_path) || !sp.Load(path_to_ansi(spm_path)).ok()) {
    cli_error("分词器加载失败: " + path_to_utf8(spm_path));
    return 3;
  }
  std::vector<int> ids;
  sp.Encode("你好，今天天气不错", &ids);
  std::vector<int64_t> in_ids(ids.begin(), ids.end());
  std::vector<int64_t> out_ids;
  {
    torch::NoGradGuard guard;
    auto input = torch::tensor(in_ids, torch::TensorOptions().dtype(torch::kInt64))
                     .reshape({1, (int64_t)in_ids.size()})
                     .to(dev);
    auto output = model->generate(input, 30, 0.8, 40);
    auto cpu = output[0].to(torch::kCPU).contiguous();
    const int64_t* d = cpu.data_ptr<int64_t>();
    out_ids.assign(d, d + cpu.numel());
  }
  std::vector<int> new_ids(out_ids.begin() + in_ids.size(), out_ids.end());
  std::string text;
  sp.Decode(new_ids, &text);
  std::cout << "[OK] 生成文本: " << text << std::endl;

  const fs::path tmp = fs::temp_directory_path() / "gpt_roundtrip.pt";
  {
    const auto sd = collect_state_dict(*model);
    save_legacy_state_dict(tmp, sd);
  }
  auto m2 = GPT(model->config);
  {
    const auto sd2 = load_legacy_state_dict(tmp, torch::kCPU);
    load_state_dict_into(*m2, sd2);
  }
  fs::remove(tmp);
  const bool same = torch::equal(m2->lm_head->weight,
                                 model->lm_head->weight.to(torch::kCPU));
  std::cout << "[OK] 保存/加载回环通过 (权重一致: " << (same ? "是" : "否") << ")"
            << std::endl;
  if (!same) return 4;
  std::cout << "SELFTEST_OK" << std::endl;
  return 0;
}

int cli_cmd_loadtest(const CliArgs& a) {
  if (a.has(L"help")) {
    cli_print_loadtest_help();
    return 0;
  }
  const std::wstring file = a.has(L"file") ? a.get(L"file") : a.pos(0);
  if (file.empty()) {
    cli_error("用法: loadtest <文件.pt>");
    return 2;
  }
  try {
    const auto sd = load_legacy_state_dict(fs::path(file), torch::kCPU);
    std::cout << "LOADTEST: 读取 " << sd.size() << " 个张量" << std::endl;
    for (const auto& [k, t] : sd) {
      std::cout << "  " << k << " dtype=" << t.scalar_type()
                << " sizes=" << t.sizes() << " sum=" << t.sum().item<double>()
                << std::endl;
    }
    std::cout << "LOADTEST_OK" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    cli_error(std::string("LOADTEST 失败: ") + e.what());
    return 1;
  }
}

} // namespace gpt
