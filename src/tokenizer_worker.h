#pragma once
// tokenizer_worker.h - 分词器训练流程（移植自 cpp分词器/src/main.cpp）
// 在进程内调用 sentencepiece 库（Trainer / Processor），无需外部 exe。
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace gpt {

// 语料统计（选择文件后即可调用，用于 GUI 实时预览自动参数）
struct CorpusStats {
  uintmax_t total_bytes = 0;
  uintmax_t unique_chars = 0;
  bool complete = true; // 大文件只采样
};

struct TokenizerTask {
  // 输入语料（为空时使用 output_dir/data/raw/corpus.txt 或目录内 *.txt）
  std::vector<std::filesystem::path> input_files;
  // 输出目录（内部结构与原版一致：models/spm.model、data/processed/token_ids.txt、
  // data/config.json、dataset.zip）
  std::filesystem::path output_dir;
  // 0 = 自动（按语料大小分档）
  int num_threads = 0;
  int vocab_size = 0;
  int input_sentence_size = 0; // 0 = 自动（按档位；档位值为 0 时表示全部句子）
  std::function<void(const std::string&)> log;
};

struct TokenizerResult {
  int num_threads = 0;
  int vocab_size = 0;
  int input_sentence_size = 0;
  std::string tier_name;
  uintmax_t corpus_bytes = 0;
  uintmax_t unique_chars = 0;
  bool charset_complete = true;
  std::filesystem::path spm_model;
  std::filesystem::path token_ids;
  std::filesystem::path config_json;
  std::filesystem::path dataset_zip;
};

// 统计一组 txt 文件（跳过不可读文件）
CorpusStats analyze_corpus(
    const std::vector<std::filesystem::path>& files,
    const std::filesystem::path& output_dir = {});

// 自动参数预览（供 GUI 在选择语料后实时显示）
struct AutoTrainPreview {
  std::string tier_name;
  int num_threads = 0;
  int vocab_size = 0;          // 已含字符集下限修正
  int input_sentence_size = 0; // 0 = 全部句子
  bool extremely_large = false;
};
AutoTrainPreview preview_auto_params(uintmax_t total_bytes, uintmax_t unique_chars,
                                     bool charset_complete);

// 执行完整流程：确定输入 → 合并 → 统计 → 自动参数 → spm_train → 分词 →
// config.json → dataset.zip。同步执行（调用方应放在工作线程）。
// 成功返回 true；失败返回 false 并写入 error。
bool run_tokenizer_task(const TokenizerTask& task, TokenizerResult* result,
                        std::string* error);

} // namespace gpt
