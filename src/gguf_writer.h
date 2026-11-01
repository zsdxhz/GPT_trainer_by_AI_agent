#pragma once
// gguf_writer.h - GGUF(v3) 写出器 + 把本项目训练的 GPT 导出为 llama.cpp 可加载的 GGUF。
//
// 本项目的 GPT 架构与 GPT-2 一致（Pre-LN、可学习位置嵌入、GELU FFN、独立 lm_head），
// 因此导出为 llama.cpp 的 `gpt2` 架构：
//   token_embd / position_embd / output_norm / output /
//   blk.N.{attn_norm, attn_qkv, attn_output, ffn_norm, ffn_up, ffn_down}(.weight/.bias)
// 词表使用 SentencePiece（tokenizer.ggml.model = "llama"），从 spm.model 直接提取，
// 因此导出的 .gguf 能被 llama.cpp / llama-server 直接加载并推理。
#include <cstdint>
#include <vector>
#include <filesystem>
#include <string>

namespace gpt {

struct GgufExportResult {
  std::filesystem::path out_path;
  std::filesystem::path spm_model; // 实际使用的分词器
  int64_t tensor_count = 0;
  int64_t vocab_size = 0;
  int64_t bytes = 0;
};

// 把 saves/modelN（config.json + model.pt）导出为 GGUF。
//   model_dir : 模型目录（含 config.json / model.pt）
//   out_path  : 输出 .gguf 路径
//   spm_model : 分词器路径；为空时自动在 model_dir 与项目内常见位置查找
bool export_gpt_to_gguf(const std::filesystem::path& model_dir,
                        const std::filesystem::path& out_path,
                        const std::filesystem::path& spm_model,
                        GgufExportResult* result, std::string* err);

// 列出项目内可用的 spm.model 候选（按优先级排序，仅返回真实存在的文件）
std::vector<std::filesystem::path> find_spm_candidates(
    const std::filesystem::path& model_dir);

// 只读 GGUF 头部的词表大小（用于给 llama.cpp 引擎挑选匹配的 SentencePiece）
bool gguf_read_vocab_size(const std::filesystem::path& gguf_path,
                          int64_t* vocab_size, std::string* err);

} // namespace gpt
