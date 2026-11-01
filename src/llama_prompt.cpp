#include "llama_prompt.h"

#include <sentencepiece_processor.h>

#include <vector>

#include "config.h"
#include "gguf_writer.h"
#include "path_util.h"

namespace fs = std::filesystem;

namespace gpt {

std::shared_ptr<sentencepiece::SentencePieceProcessor> load_spm_for_gguf(
    const fs::path& gguf_path) {
  int64_t vocab = 0;
  std::string err;
  if (!gguf_read_vocab_size(gguf_path, &vocab, &err) || vocab <= 0) return nullptr;

  for (const auto& cand : find_spm_candidates(fs::path())) {
    auto sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
    if (!sp->Load(path_to_ansi(cand)).ok()) continue;
    if ((int64_t)sp->GetPieceSize() != vocab) continue;
    return sp;
  }
  return nullptr;
}

// token 通道：先用 sentencepiece 编码，过长时保留最近 max_seq_len 个 token
// 返回 false 且 err 为空 = 编码失败（调用方可回退文本通道）
static bool complete_with_tokens(LlamaEngine& engine, const std::string& prompt,
                                 const sentencepiece::SentencePieceProcessor* sp,
                                 int64_t max_seq_len,
                                 const LlamaSamplingParams& params,
                                 const LlamaEngine::DeltaFn& on_delta,
                                 std::string* out_text, std::string* err,
                                 size_t* prompt_tokens) {
  std::vector<int> ids;
  const auto status = sp->Encode(prompt, &ids);
  if (!status.ok() || ids.empty()) {
    if (err) err->clear();
    return false;
  }
  if (max_seq_len > 1 && (int64_t)ids.size() > max_seq_len) {
    ids.erase(ids.begin(), ids.begin() + (ids.size() - (size_t)max_seq_len));
  }
  if (prompt_tokens != nullptr) *prompt_tokens = ids.size();
  std::vector<int32_t> token_ids(ids.begin(), ids.end());
  return engine.complete_tokens(token_ids, params, on_delta, out_text, err);
}

// 文本/对话通道没有本地分词器：按 UTF-8 字节数粗估 token 数（中文约 2.5 字节/token）
static size_t estimate_tokens(const std::string& text) {
  return std::max<size_t>(1, (size_t)(text.size() / 2.5));
}

bool llama_complete_smart(LlamaEngine& engine, const std::string& prompt,
                          const sentencepiece::SentencePieceProcessor* sp,
                          int64_t max_seq_len,
                          const LlamaSamplingParams& params,
                          const LlamaEngine::DeltaFn& on_delta,
                          std::string* out_text, std::string* err,
                          LlamaPromptMode mode,
                          const std::string& system_prompt,
                          size_t* prompt_tokens) {
  auto set_prompt_tokens = [&](size_t n) {
    if (prompt_tokens != nullptr) *prompt_tokens = n;
  };
  auto text_channel = [&]() {
    set_prompt_tokens(estimate_tokens(system_prompt + prompt));
    return engine.complete(prompt, params, on_delta, out_text, err);
  };

  switch (mode) {
    case LlamaPromptMode::kChat:
      set_prompt_tokens(estimate_tokens(system_prompt + prompt));
      return engine.complete_chat(system_prompt, prompt, params, on_delta,
                                  out_text, err);
    case LlamaPromptMode::kText:
      return text_channel();
    case LlamaPromptMode::kTokens:
      if (sp == nullptr) return text_channel();
      if (complete_with_tokens(engine, prompt, sp, max_seq_len, params, on_delta,
                               out_text, err, prompt_tokens)) {
        return true;
      }
      if (err != nullptr && !err->empty()) return false; // 推理真失败
      return text_channel();
    case LlamaPromptMode::kAuto:
    default:
      break;
  }

  // ---- 自动模式 ----
  // 1) 本项目自己训练的模型：词表能被 spm 匹配 → 用 token 通道（中文最稳）
  if (sp != nullptr) {
    if (complete_with_tokens(engine, prompt, sp, max_seq_len, params, on_delta,
                             out_text, err, prompt_tokens)) {
      return true;
    }
    if (err != nullptr && !err->empty()) return false;
    return text_channel();
  }
  // 2) 外部对话模型（Qwen/Llama-Instruct…）：自带 chat_template → 对话通道，
  //    由 llama.cpp 套用模型原生格式（ChatML 等），指令遵循效果最好
  if (engine.has_chat_template()) {
    set_prompt_tokens(estimate_tokens(system_prompt + prompt));
    if (engine.complete_chat(system_prompt, prompt, params, on_delta, out_text,
                             err)) {
      return true;
    }
    // 对话通道失败（模板/分词不兼容等）→ 自动退回文本通道再试一次
    if (out_text != nullptr) out_text->clear();
    if (err != nullptr) err->clear();
    return text_channel();
  }
  // 3) 其余 GGUF：文本通道（llama.cpp 自带分词器）
  return text_channel();
}

} // namespace gpt
