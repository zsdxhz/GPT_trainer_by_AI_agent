// tc2sc.cpp — 繁体中文 → 简体中文 批量转换（OpenCC t2s 字典驱动）
//
// 字典：TSPhrases.txt（短语，最长匹配优先）+ TSCharacters.txt（单字）
// 与 OpenCC 的 t2s 配置等价：先短语、后单字，未命中则原样输出。
// 采用 UTF-8 流式处理，内存占用恒定，可处理数 GB 语料。
//
// 用法：
//   tc2sc.exe <TSPhrases.txt> <TSCharacters.txt> <输入文件|目录> <输出> [--inplace]
//     - 输入是文件：输出为文件（--inplace 时写临时文件再替换原文件）
//     - 输入是目录：把目录下所有 *.txt 转换到 <输出目录>（--inplace 时原地替换）
//   示例：
//     tc2sc.exe TSPhrases.txt TSCharacters.txt wiki_pure.txt wiki_pure.sc.txt
//
// 字典版权：OpenCC (Apache License 2.0) https://github.com/BYVoid/OpenCC
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// UTF-8 编解码
// ---------------------------------------------------------------------------
inline void append_utf8(std::string* out, uint32_t cp) {
  if (cp < 0x80) {
    out->push_back((char)cp);
  } else if (cp < 0x800) {
    out->push_back((char)(0xC0 | (cp >> 6)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back((char)(0xE0 | (cp >> 12)));
    out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out->push_back((char)(0xF0 | (cp >> 18)));
    out->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// 从 buf 解码一个码点；返回消耗的字节数，0 表示不完整/非法
inline size_t decode_utf8(const char* buf, size_t avail, uint32_t* cp) {
  if (avail == 0) return 0;
  const unsigned char c = (unsigned char)buf[0];
  if (c < 0x80) { *cp = c; return 1; }
  if ((c & 0xE0) == 0xC0) {
    if (avail < 2) return 0;
    if (((unsigned char)buf[1] & 0xC0) != 0x80) return 0;
    *cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)buf[1] & 0x3F);
    return 2;
  }
  if ((c & 0xF0) == 0xE0) {
    if (avail < 3) return 0;
    if (((unsigned char)buf[1] & 0xC0) != 0x80 || ((unsigned char)buf[2] & 0xC0) != 0x80) return 0;
    *cp = ((uint32_t)(c & 0x0F) << 12) | (((unsigned char)buf[1] & 0x3F) << 6) |
          ((unsigned char)buf[2] & 0x3F);
    return 3;
  }
  if ((c & 0xF8) == 0xF0) {
    if (avail < 4) return 0;
    if (((unsigned char)buf[1] & 0xC0) != 0x80 || ((unsigned char)buf[2] & 0xC0) != 0x80 ||
        ((unsigned char)buf[3] & 0xC0) != 0x80)
      return 0;
    *cp = ((uint32_t)(c & 0x07) << 18) | (((unsigned char)buf[1] & 0x3F) << 12) |
          (((unsigned char)buf[2] & 0x3F) << 6) | ((unsigned char)buf[3] & 0x3F);
    return 4;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// 字典
// ---------------------------------------------------------------------------
struct TrieNode {
  std::vector<std::pair<uint32_t, int32_t>> next;  // 码点 -> 子节点（有序，顺序查找）
  int32_t repl = -1;                               // 替换串在 repl_ 池中的下标
};

struct Stats {
  uint64_t cps = 0;
  uint64_t char_conv = 0;    // 单字替换次数
  uint64_t phrase_conv = 0;  // 短语替换次数
  uint64_t phrase_cps = 0;   // 被短语吃掉的码点数
};

class Converter {
 public:
  bool load_phrases(const wchar_t* path);
  bool load_chars(const wchar_t* path);

  // 转换 cps[begin, end) 追加到 out
  void convert(const std::vector<uint32_t>& cps, size_t begin, size_t end,
               std::string* out, Stats* st) const {
    size_t i = begin;
    while (i < end) {
      // 最长匹配短语
      int32_t node = 0;
      int32_t best = -1;
      size_t best_len = 0;
      size_t j = i;
      while (j < end) {
        const auto& kids = trie_[node].next;
        int32_t child = -1;
        for (const auto& kv : kids) {
          if (kv.first == cps[j]) { child = kv.second; break; }
        }
        if (child < 0) break;
        node = child;
        ++j;
        if (trie_[node].repl >= 0) { best = trie_[node].repl; best_len = j - i; }
      }
      if (best >= 0) {
        out->append(repl_[(size_t)best]);
        i += best_len;
        st->phrase_conv++;
        st->phrase_cps += best_len;
        st->cps += best_len;
        continue;
      }
      // 单字
      const uint32_t cp = cps[i];
      ++st->cps;
      if (cp < 0x80) {
        out->push_back((char)cp);
        ++i;
        continue;
      }
      const auto it = char_map_.find(cp);
      if (it != char_map_.end()) {
        if (it->second < 0x80) {
          out->push_back((char)it->second);
        } else {
          append_utf8(out, (uint32_t)it->second);
        }
        st->char_conv++;
      } else {
        const auto mit = char_multi_.find(cp);
        if (mit != char_multi_.end()) {
          out->append(mit->second);
          st->char_conv++;
        } else {
          append_utf8(out, cp);
        }
      }
      ++i;
    }
  }

  size_t max_phrase_cps() const { return max_phrase_cps_; }
  size_t phrase_count() const { return phrases_; }
  size_t char_count() const { return char_map_.size() + char_multi_.size(); }

 private:
  bool insert_phrase(const std::string& key, const std::string& value);

  std::vector<TrieNode> trie_{1};
  std::vector<std::string> repl_;
  std::unordered_map<uint32_t, uint32_t> char_map_;
  std::unordered_map<uint32_t, std::string> char_multi_;
  size_t max_phrase_cps_ = 1;
  size_t phrases_ = 0;
};

// "k<TAB>v1 v2 v3" → first candidate
std::string first_candidate(const std::string& values) {
  if (values.empty()) return {};
  const size_t sp = values.find(' ');
  return sp == std::string::npos ? values : values.substr(0, sp);
}

// 去掉行尾的 \r\n（fgets 会保留换行符，带上会把换行写进替换结果）
std::string trim_cr(const std::string& s) {
  std::string t = s;
  while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
  return t;
}

bool Converter::insert_phrase(const std::string& key, const std::string& value) {
  std::vector<uint32_t> cps;
  size_t pos = 0;
  while (pos < key.size()) {
    uint32_t cp = 0;
    const size_t n = decode_utf8(key.data() + pos, key.size() - pos, &cp);
    if (n == 0) return false;
    cps.push_back(cp);
    pos += n;
  }
  if (cps.size() < 2) return false;
  int32_t node = 0;
  for (uint32_t cp : cps) {
    int32_t child = -1;
    for (const auto& kv : trie_[(size_t)node].next) {
      if (kv.first == cp) { child = kv.second; break; }
    }
    if (child < 0) {
      child = (int32_t)trie_.size();
      trie_.emplace_back();
      trie_[(size_t)node].next.emplace_back(cp, child);
    }
    node = child;
  }
  repl_.push_back(value);
  trie_[(size_t)node].repl = (int32_t)(repl_.size() - 1);
  max_phrase_cps_ = std::max(max_phrase_cps_, cps.size());
  ++phrases_;
  return true;
}

bool Converter::load_phrases(const wchar_t* path) {
  FILE* f = nullptr;
  if (_wfopen_s(&f, path, L"rb") != 0 || f == nullptr) {
    std::fprintf(stderr, "无法打开短语字典\n");
    return false;
  }
  std::string line;
  char buf[65536];
  while (std::fgets(buf, sizeof(buf), f) != nullptr) {
    line = trim_cr(buf);
    const size_t tab = line.find('\t');
    if (tab == std::string::npos || tab == 0) continue;
    const std::string key = line.substr(0, tab);
    const std::string value = first_candidate(line.substr(tab + 1));
    if (value.empty()) continue;
    insert_phrase(key, value);
  }
  std::fclose(f);
  return true;
}

bool Converter::load_chars(const wchar_t* path) {
  FILE* f = nullptr;
  if (_wfopen_s(&f, path, L"rb") != 0 || f == nullptr) {
    std::fprintf(stderr, "无法打开单字字典\n");
    return false;
  }
  std::string line;
  char buf[65536];
  while (std::fgets(buf, sizeof(buf), f) != nullptr) {
    line = trim_cr(buf);
    const size_t tab = line.find('\t');
    if (tab == std::string::npos || tab == 0) continue;
    const std::string key = line.substr(0, tab);
    const std::string value = first_candidate(line.substr(tab + 1));
    if (value.empty()) continue;
    uint32_t cp = 0;
    const size_t n = decode_utf8(key.data(), key.size(), &cp);
    if (n == 0 || n != key.size()) continue;  // 只处理单字键
    uint32_t vcp = 0;
    const size_t vn = decode_utf8(value.data(), value.size(), &vcp);
    if (vn != 0 && vn == value.size()) {
      char_map_[cp] = vcp;
    } else {
      char_multi_[cp] = value;
    }
  }
  std::fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// 流式转换
// ---------------------------------------------------------------------------
bool convert_file(Converter& conv, const fs::path& in_path, const fs::path& out_path,
                  Stats* st, std::string* err) {
  FILE* in = nullptr;
  if (_wfopen_s(&in, in_path.wstring().c_str(), L"rb") != 0 || in == nullptr) {
    *err = "无法读取输入文件";
    return false;
  }
  FILE* out = nullptr;
  if (_wfopen_s(&out, out_path.wstring().c_str(), L"wb") != 0 || out == nullptr) {
    std::fclose(in);
    *err = "无法写入输出文件";
    return false;
  }

  const size_t kChunk = 8u << 20;
  std::vector<char> rbuf(kChunk);
  std::string pending;
  pending.reserve(kChunk * 2);
  std::string obuf;
  obuf.reserve(kChunk * 2);
  std::vector<uint32_t> cps;
  const size_t keep_cps = conv.max_phrase_cps() > 1 ? conv.max_phrase_cps() - 1 : 0;

  uint64_t bytes_out = 0;
  auto flush = [&]() {
    if (!obuf.empty()) {
      std::fwrite(obuf.data(), 1, obuf.size(), out);
      bytes_out += obuf.size();
      obuf.clear();
    }
  };

  for (;;) {
    const size_t got = std::fread(rbuf.data(), 1, rbuf.size(), in);
    const bool eof = (got == 0) || (got < rbuf.size() && std::feof(in) != 0);
    pending.append(rbuf.data(), got);
    // 解码完整码点
    cps.clear();
    size_t pos = 0;
    while (pos < pending.size()) {
      uint32_t cp = 0;
      const size_t n = decode_utf8(pending.data() + pos, pending.size() - pos, &cp);
      if (n == 0) break;
      cps.push_back(cp);
      pos += n;
    }
    const size_t decoded_bytes = pos;
    size_t safe_cps = cps.size();
    if (!eof && cps.size() > keep_cps) safe_cps = cps.size() - keep_cps;
    size_t safe_bytes = 0;
    {
      size_t p = 0;
      for (size_t i = 0; i < safe_cps; ++i) {
        uint32_t cp = 0;
        p += decode_utf8(pending.data() + p, pending.size() - p, &cp);
      }
      safe_bytes = p;
    }
    conv.convert(cps, 0, safe_cps, &obuf, st);
    if (obuf.size() > (1u << 20)) flush();
    pending.erase(0, safe_bytes);
    (void)decoded_bytes;
    if (eof) break;
  }
  flush();
  std::fclose(in);
  std::fclose(out);
  (void)bytes_out;
  return true;
}

} // namespace

int wmain(int argc, wchar_t** argv) {
  if (argc < 5) {
    std::printf(
        "用法: tc2sc.exe <TSPhrases.txt> <TSCharacters.txt> <输入文件|目录> <输出> [--inplace]\n");
    return 2;
  }
  Converter conv;
  if (!conv.load_phrases(argv[1]) || !conv.load_chars(argv[2])) return 1;
  std::printf("字典已载入: 短语 %zu 条（最长 %zu 字）, 单字 %zu 个\n", conv.phrase_count(),
              conv.max_phrase_cps(), conv.char_count());

  const fs::path in_path = argv[3];
  const fs::path out_path = argv[4];
  const bool inplace = (argc >= 6 && std::wstring(argv[5]) == L"--inplace");

  Stats total;
  std::string err;
  if (fs::is_directory(in_path)) {
    std::error_code ec;
    fs::create_directories(out_path, ec);
    size_t ok = 0, fail = 0;
    for (const auto& entry : fs::directory_iterator(in_path, ec)) {
      if (!entry.is_regular_file(ec) || entry.path().extension() != ".txt") continue;
      const fs::path dst = out_path / entry.path().filename();
      Stats st;
      if (convert_file(conv, entry.path(), dst, &st, &err)) {
        ++ok;
        total.cps += st.cps;
        total.char_conv += st.char_conv;
        total.phrase_conv += st.phrase_conv;
        total.phrase_cps += st.phrase_cps;
        std::printf("  OK  %-48s 单字 %llu  短语 %llu\n",
                    entry.path().filename().string().c_str(),
                    (unsigned long long)st.char_conv, (unsigned long long)st.phrase_conv);
      } else {
        ++fail;
        std::fprintf(stderr, "  失败 %s: %s\n", entry.path().filename().string().c_str(),
                     err.c_str());
      }
    }
    std::printf("目录转换完成: 成功 %zu, 失败 %zu\n", ok, fail);
    return fail == 0 ? 0 : 1;
  }

  fs::path tmp_out = out_path;
  if (inplace) tmp_out = in_path.wstring() + L".sc.tmp";
  if (!convert_file(conv, in_path, tmp_out, &total, &err)) {
    std::fprintf(stderr, "转换失败: %s\n", err.c_str());
    return 1;
  }
  if (inplace) {
    std::error_code ec;
    fs::rename(tmp_out, in_path, ec);
    if (ec) {
      std::fprintf(stderr, "替换原文件失败: %s\n", ec.message().c_str());
      return 1;
    }
  }
  std::printf("完成: %s -> %s\n  码点 %llu, 单字替换 %llu, 短语替换 %llu（覆盖 %llu 字）\n",
              in_path.string().c_str(), out_path.string().c_str(),
              (unsigned long long)total.cps, (unsigned long long)total.char_conv,
              (unsigned long long)total.phrase_conv, (unsigned long long)total.phrase_cps);
  return 0;
}
