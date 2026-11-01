#pragma once
// llama_prompt.h - llama.cpp 提示词通道选择
//
// 背景：本项目导出的 GGUF 使用中文 SentencePiece 词表（无 byte-fallback），
// llama.cpp 自带分词器遇到拉丁字符会返回 400（tokenize: unordered_map::at）。
// 因此对这类模型改为「先用 sentencepiece 分词 → 走 llama.cpp 的 token 通道」，
// 词表里没有的字符会变成 <unk>，与训练时完全一致；其它（外部）GGUF 继续用文本通道。
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "llama_engine.h"

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace gpt {

// 提示词通道：
//   kAuto   - 自动：本项目模型（匹配到 spm）走 token 通道；外部对话模型（自带
//             chat_template）走对话通道；其余走文本通道
//   kText   - 强制文本通道（llama.cpp 自带分词器）
//   kTokens - 强制 token 通道（先 sentencepiece 编码）
//   kChat   - 强制对话通道（/v1/chat/completions，套用模型 chat_template）
enum class LlamaPromptMode { kAuto = 0, kText, kTokens, kChat };

// 读取 GGUF 的词表大小，并在项目内挑选词表大小一致的 spm.model 载入。
// 找不到时返回 nullptr（调用方回退到文本通道）。
std::shared_ptr<sentencepiece::SentencePieceProcessor> load_spm_for_gguf(
    const std::filesystem::path& gguf_path);

// 依据 sp 与 mode 自动选择通道：
//   sp != null → token 通道（先 sentencepiece 编码，过长时保留最近 max_seq_len 个 token）
//   sp == null 且模型自带 chat_template → 对话通道
//   否则 → 文本通道
bool llama_complete_smart(LlamaEngine& engine, const std::string& prompt,
                          const sentencepiece::SentencePieceProcessor* sp,
                          int64_t max_seq_len,
                          const LlamaSamplingParams& params,
                          const LlamaEngine::DeltaFn& on_delta,
                          std::string* out_text, std::string* err,
                          LlamaPromptMode mode = LlamaPromptMode::kAuto,
                          const std::string& system_prompt = {},
                          size_t* prompt_tokens = nullptr);

} // namespace gpt
