#include "tokenizer_worker.h"

#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "json_util.h"
#include "path_util.h"
#include "zip_util.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

// ---------------------------------------------------------------------------
// 文件查找 / 合并（与原版一致）
// ---------------------------------------------------------------------------
std::vector<fs::path> get_txt_files(const fs::path& dir) {
  std::vector<fs::path> files;
  std::error_code ec;
  if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return files;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    if (entry.is_regular_file(ec) && entry.path().extension() == ".txt" &&
        entry.path().filename() != "_merged_temp.txt") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

uintmax_t sum_file_sizes(const std::vector<fs::path>& files) {
  uintmax_t total = 0;
  for (const auto& f : files) {
    std::error_code ec;
    uintmax_t sz = fs::file_size(f, ec);
    if (!ec) total += sz;
  }
  return total;
}

void merge_txt_files(const std::vector<fs::path>& files,
                     const fs::path& output) {
  if (fs::exists(output)) fs::remove(output);
  std::ofstream out(output, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("无法创建合并文件: " + path_to_utf8(output));
  }
  for (const auto& f : files) {
    if (f.filename() == "_merged_temp.txt") continue;
    std::ifstream in(f, std::ios::binary);
    if (!in.is_open()) continue;
    out << in.rdbuf();
    out << '\n';
  }
}

// ---------------------------------------------------------------------------
// 唯一字符数统计（大文件 8 段均匀采样，与原版一致）
// ---------------------------------------------------------------------------
void count_utf8_codepoints(const char* p, size_t n,
                           std::unordered_set<uint32_t>& chars) {
  const char* end = p + n;
  while (p < end) {
    unsigned char c = static_cast<unsigned char>(*p);
    int len = 0;
    uint32_t cp = 0;
    if (c < 0x80) { chars.insert(c); ++p; continue; }
    else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
    else { ++p; continue; }
    if (p + len > end) break;
    bool ok = true;
    for (int i = 1; i < len; ++i) {
      unsigned char cc = static_cast<unsigned char>(p[i]);
      if ((cc & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (cc & 0x3F);
    }
    if (ok) chars.insert(cp);
    p += len;
  }
}

uintmax_t count_unique_chars(const fs::path& file, bool* complete = nullptr,
                             uintmax_t sample_limit = 64ULL * 1024 * 1024) {
  std::ifstream in(file, std::ios::binary);
  if (!in.is_open()) return 0;

  std::error_code ec;
  const uintmax_t file_size = fs::file_size(file, ec);
  if (ec) return 0;

  std::unordered_set<uint32_t> chars;
  chars.reserve(65536);

  auto scan_range = [&](uintmax_t offset, uintmax_t len) {
    in.seekg(static_cast<std::streamoff>(offset));
    constexpr size_t kChunk = 1 << 20;
    std::string buf(kChunk, '\0');
    uintmax_t done = 0;
    while (done < len && in) {
      size_t want = static_cast<size_t>(std::min<uintmax_t>(kChunk, len - done));
      in.read(&buf[0], static_cast<std::streamsize>(want));
      std::streamsize got = in.gcount();
      if (got <= 0) break;
      count_utf8_codepoints(buf.data(), static_cast<size_t>(got), chars);
      done += static_cast<uintmax_t>(got);
    }
  };

  bool full = true;
  if (file_size <= sample_limit) {
    scan_range(0, file_size);
  } else {
    constexpr int kSegments = 8;
    const uintmax_t seg_len = sample_limit / kSegments;
    for (int i = 0; i < kSegments; ++i) {
      uintmax_t off = i * (file_size - seg_len) / (kSegments - 1);
      scan_range(off, seg_len);
    }
    full = false;
  }
  if (complete) *complete = full;
  return chars.size();
}

// ---------------------------------------------------------------------------
// 自动参数分档（与原版一致）
// ---------------------------------------------------------------------------
struct TrainParams {
  std::string tier_name;
  int num_threads;
  int vocab_size;
  int input_sentence_size;
  bool extremely_large;
};

TrainParams auto_params(uintmax_t total_bytes) {
  static const struct Tier {
    uintmax_t max_bytes;
    int vocab_size;
    int input_sentence_size;
    unsigned max_threads;
    const char* name;
    bool extremely_large;
  } tiers[] = {
      { 1ULL * 1024 * 1024, 2000, 0, 4, "小型语料 (< 1 MB)", false },
      { 10ULL * 1024 * 1024, 4000, 0, 8, "小型语料 (1 - 10 MB)", false },
      { 100ULL * 1024 * 1024, 8000, 300000, 16, "中型语料 (10 - 100 MB)", false },
      { 500ULL * 1024 * 1024, 16000, 1000000, 32, "大型语料 (100 - 500 MB)", false },
      { UINTMAX_MAX, 32000, 2000000, 0, "超大型语料 (>= 500 MB)", true },
  };
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned cores = (hw == 0) ? 4u : hw;
  for (const auto& t : tiers) {
    if (total_bytes < t.max_bytes) {
      unsigned nt = (t.max_threads == 0) ? cores : std::min(cores, t.max_threads);
      return { t.name, static_cast<int>(nt), t.vocab_size,
               t.input_sentence_size, t.extremely_large };
    }
  }
  return { tiers[4].name, static_cast<int>(cores), tiers[4].vocab_size,
           tiers[4].input_sentence_size, tiers[4].extremely_large };
}

void log_line(const TokenizerTask& task, const std::string& s) {
  if (task.log) task.log(s);
}

} // namespace

// ---------------------------------------------------------------------------
// 统计 / 自动参数预览
// ---------------------------------------------------------------------------
CorpusStats analyze_corpus(const std::vector<fs::path>& files,
                           const fs::path& output_dir) {
  CorpusStats st;
  std::vector<fs::path> list = files;
  if (list.empty() && !output_dir.empty()) {
    const fs::path corpus_txt = output_dir / "data" / "raw" / "corpus.txt";
    if (fs::exists(corpus_txt)) {
      list.push_back(corpus_txt);
    } else {
      list = get_txt_files(output_dir / "data" / "raw");
    }
  }
  st.total_bytes = sum_file_sizes(list);
  // 字符集统计：逐文件统计求和（多文件时略偏高，仅用于预览；
  // 训练流程内部对合并后的单一输入精确统计）
  uintmax_t chars = 0;
  bool complete = true;
  for (const auto& f : list) {
    bool c = true;
    chars += count_unique_chars(f, &c);
    if (!c) complete = false;
  }
  st.unique_chars = chars;
  st.complete = complete;
  return st;
}

AutoTrainPreview preview_auto_params(uintmax_t total_bytes, uintmax_t unique_chars,
                                     bool charset_complete) {
  const TrainParams auto_cfg = auto_params(total_bytes);
  const uintmax_t margin = charset_complete ? 512u : 1024u;
  const int vocab_floor =
      static_cast<int>(std::min<uintmax_t>(unique_chars + margin, 100000u));
  AutoTrainPreview p;
  p.tier_name = auto_cfg.tier_name;
  p.num_threads = auto_cfg.num_threads;
  p.vocab_size = std::max(auto_cfg.vocab_size, vocab_floor);
  p.input_sentence_size = auto_cfg.input_sentence_size;
  p.extremely_large = auto_cfg.extremely_large;
  return p;
}

// ---------------------------------------------------------------------------
// 完整流程
// ---------------------------------------------------------------------------
bool run_tokenizer_task(const TokenizerTask& task, TokenizerResult* result,
                        std::string* error) {
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    log_line(task, "失败: " + msg);
    return false;
  };

  try {
    const fs::path out = task.output_dir;
    fs::create_directories(out);
    fs::create_directories(out / "data" / "raw");
    fs::create_directories(out / "data" / "processed");
    fs::create_directories(out / "models");

    // ---------- 1. 确定输入 ----------
    std::vector<fs::path> txt_files = task.input_files;
    bool from_selection = !txt_files.empty();
    if (from_selection) {
      std::sort(txt_files.begin(), txt_files.end());
      log_line(task, "已选择 " + std::to_string(txt_files.size()) + " 个 txt 文件:");
      for (const auto& f : txt_files) log_line(task, "  - " + path_to_utf8(f));
    }

    fs::path input_file;
    uintmax_t total_bytes = 0;
    bool need_merge = false;

    if (!from_selection) {
      const fs::path corpus_txt = out / "data" / "raw" / "corpus.txt";
      if (fs::exists(corpus_txt)) {
        input_file = corpus_txt;
        std::error_code ec;
        total_bytes = fs::file_size(input_file, ec);
        if (ec) total_bytes = 0;
        log_line(task, "使用 corpus.txt 作为输入");
      } else {
        txt_files = get_txt_files(out / "data" / "raw");
        if (txt_files.empty()) {
          return fail("data/raw/ 下没有 .txt 文件，请先选择语料");
        }
      }
    }

    if (input_file.empty()) {
      if (txt_files.size() == 1) {
        input_file = txt_files[0];
        std::error_code ec;
        total_bytes = fs::file_size(input_file, ec);
        if (ec) total_bytes = 0;
        log_line(task, "使用唯一 txt 文件: " + path_to_utf8(input_file));
      } else {
        need_merge = true;
        total_bytes = sum_file_sizes(txt_files);
        input_file = out / "data" / "raw" / "_merged_temp.txt";
        log_line(task, "检测到 " + std::to_string(txt_files.size()) +
                           " 个 txt 文件，将先合并再训练。");
      }
    }

    if (total_bytes == 0) {
      return fail("输入文件为空（0 字节），无法训练！");
    }

    // ---------- 2. 合并 ----------
    if (need_merge) {
      log_line(task, "合并 " + std::to_string(txt_files.size()) + " 个 txt 文件...");
      merge_txt_files(txt_files, input_file);
      log_line(task, "合并完成: " + path_to_utf8(input_file));
    }

    // ---------- 3. 统计 ----------
    bool charset_complete = true;
    const uintmax_t unique_chars =
        count_unique_chars(input_file, &charset_complete);
    const uintmax_t margin = charset_complete ? 512u : 1024u;
    const int vocab_floor =
        static_cast<int>(std::min<uintmax_t>(unique_chars + margin, 100000u));

    // ---------- 4. 自动参数 ----------
    const TrainParams auto_cfg = auto_params(total_bytes);
    int num_threads = task.num_threads > 0 ? task.num_threads : auto_cfg.num_threads;
    int vocab_size =
        task.vocab_size > 0 ? task.vocab_size : auto_cfg.vocab_size;
    int input_sentence_size = task.input_sentence_size > 0
                                  ? task.input_sentence_size
                                  : auto_cfg.input_sentence_size;
    // 自动模式下词表不得低于档位默认值，且不得低于字符集下限
    if (task.vocab_size <= 0) vocab_size = std::max(auto_cfg.vocab_size, vocab_floor);
    if (vocab_size < vocab_floor) {
      log_line(task, "词表大小 " + std::to_string(vocab_size) +
                         " 低于语料字符集所需下限 " + std::to_string(vocab_floor) +
                         "，已自动提升为 " + std::to_string(vocab_floor));
      vocab_size = vocab_floor;
    }

    const double total_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    log_line(task, "==================================================");
    log_line(task, " 语料总大小: " + std::to_string(total_bytes) + " 字节 (" +
                       std::to_string(total_mb) + " MB)");
    log_line(task, " 档位判定  : " + auto_cfg.tier_name);
    log_line(task, " 字符集    : " + std::to_string(unique_chars) + " 个不同字符" +
                       (charset_complete ? "" : " (采样)"));
    log_line(task, " 训练参数  : 线程数 " + std::to_string(num_threads) +
                       " | 词表大小 " + std::to_string(vocab_size) +
                       " | 最大句子数 " +
                       (input_sentence_size > 0
                            ? std::to_string(input_sentence_size)
                            : std::string("全部")));
    if (auto_cfg.extremely_large) {
      log_line(task, " 训练模式  : 超大语料增量模式 (内存受限，可全量训练)");
    }
    log_line(task, "==================================================");
    if (total_bytes > 100 * 1024 * 1024) {
      log_line(task, "警告: 输入文件较大，训练可能耗时较长且内存占用较高");
    }

    // ---------- 5. 训练（进程内 sentencepiece） ----------
    const fs::path model_prefix = out / "models" / "spm";
    std::string train_args =
        "--input=" + path_to_ansi(input_file) +
        " --model_prefix=" + path_to_ansi(model_prefix) +
        " --vocab_size=" + std::to_string(vocab_size) +
        " --model_type=unigram --character_coverage=1.0 --num_threads=" +
        std::to_string(num_threads) + " --shuffle_input_sentence=false";
    if (input_sentence_size > 0) {
      train_args += " --input_sentence_size=" + std::to_string(input_sentence_size);
    }
    if (auto_cfg.extremely_large) {
      train_args += " --train_extremely_large_corpus=true";
    }

    log_line(task, "开始训练 SentencePiece (unigram)...");
    log_line(task, "参数: " + train_args);
    const auto status = sentencepiece::SentencePieceTrainer::Train(train_args);
    if (!status.ok()) {
      return fail("训练失败: " + status.ToString());
    }
    log_line(task, "训练完成。");

    // ---------- 6. 分词生成 token_ids.txt ----------
    log_line(task, "分词中...");
    sentencepiece::SentencePieceProcessor sp;
    const auto load_status = sp.Load(path_to_ansi(model_prefix.string() + ".model"));
    if (!load_status.ok()) {
      return fail("加载分词模型失败: " + load_status.ToString());
    }
    {
      std::ifstream in(input_file, std::ios::binary);
      std::ofstream oids(out / "data" / "processed" / "token_ids.txt",
                         std::ios::binary | std::ios::trunc);
      if (!oids.is_open()) {
        return fail("无法创建 token_ids.txt");
      }
      std::string line;
      uintmax_t lines = 0;
      while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<int> ids;
        sp.Encode(line, &ids);
        if (ids.empty()) {
          oids << '\n';
        } else {
          for (size_t i = 0; i < ids.size(); ++i) {
            if (i) oids << ' ';
            oids << ids[i];
          }
          oids << '\n';
        }
        ++lines;
      }
      log_line(task, "分词完成: " + std::to_string(lines) + " 行");
    }

    // ---------- 7. config.json ----------
    const fs::path config_json = out / "data" / "config.json";
    {
      Json::Object obj;
      obj["vocab_size"] = Json(vocab_size);
      obj["max_seq_len"] = Json(256);
      obj["model_type"] = Json(std::string("unigram"));
      Json cfg(std::move(obj));
      std::ofstream f(config_json);
      f << cfg.dump(2);
      log_line(task, "已生成 config.json");
    }

    // ---------- 8. 打包 dataset.zip ----------
    const fs::path dataset_zip = out / "dataset.zip";
    std::string zerr;
    if (!create_zip(dataset_zip,
                    {{"spm.model", model_prefix.string() + ".model"},
                     {"token_ids.txt",
                      (out / "data" / "processed" / "token_ids.txt").string()},
                     {"config.json", config_json.string()}},
                    &zerr)) {
      log_line(task, "打包失败（不影响模型文件）: " + zerr);
    } else {
      log_line(task, "打包成功: " + path_to_utf8(dataset_zip));
    }

    // ---------- 9. 清理 ----------
    if (input_file.filename() == "_merged_temp.txt" && fs::exists(input_file)) {
      fs::remove(input_file);
      log_line(task, "已清理临时合并文件。");
    }

    log_line(task, "所有操作完成！");

    if (result) {
      result->num_threads = num_threads;
      result->vocab_size = vocab_size;
      result->input_sentence_size = input_sentence_size;
      result->tier_name = auto_cfg.tier_name;
      result->corpus_bytes = total_bytes;
      result->unique_chars = unique_chars;
      result->charset_complete = charset_complete;
      result->spm_model = model_prefix.string() + ".model";
      result->token_ids = out / "data" / "processed" / "token_ids.txt";
      result->config_json = config_json;
      result->dataset_zip = dataset_zip;
    }
    return true;
  } catch (const std::exception& e) {
    return fail(std::string("异常: ") + e.what());
  }
}

} // namespace gpt
