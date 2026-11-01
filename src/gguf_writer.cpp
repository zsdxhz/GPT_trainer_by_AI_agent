#include "gguf_writer.h"

#include <sentencepiece_processor.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "config.h"
#include "model.h"
#include "json_util.h"
#include "legacy_serial.h"
#include "path_util.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

// ---------------------------------------------------------------------------
// GGUF 常量
// ---------------------------------------------------------------------------
constexpr uint32_t kGgufMagic = 0x46554747u; // "GGUF"
constexpr uint32_t kGgufVersion = 3;
constexpr uint64_t kGgufAlignment = 32;

enum GgufType : uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12,
};

enum GgmlType : uint32_t {
  kGgmlF32 = 0,
  kGgmlF16 = 1,
};

// 一个元数据键值
struct GgufValue {
  enum class Kind { U32, I32, F32, Bool, String, StringArray, F32Array, I32Array };
  Kind kind = Kind::U32;
  uint32_t u32 = 0;
  int32_t i32 = 0;
  float f32 = 0.0f;
  bool b = false;
  std::string str;
  std::vector<std::string> strings;
  std::vector<float> f32s;
  std::vector<int32_t> i32s;
};

struct GgufTensor {
  std::string name;
  std::vector<uint64_t> dims; // 最内层维度在前（llama.cpp 的 ne[] 顺序）
  uint32_t type = kGgmlF32;
  torch::Tensor data; // CPU, 连续
  uint64_t offset = 0;
  uint64_t nbytes = 0;
};

// ---------------------------------------------------------------------------
// 二进制写出
// ---------------------------------------------------------------------------
class BinaryWriter {
 public:
  explicit BinaryWriter(const fs::path& path) : out_(path, std::ios::binary) {
    if (!out_) throw std::runtime_error("无法写入文件: " + path_to_utf8(path));
  }

  void u8(uint8_t v) { out_.write(reinterpret_cast<const char*>(&v), 1); }
  void u16(uint16_t v) { raw(&v, sizeof(v)); }
  void u32(uint32_t v) { raw(&v, sizeof(v)); }
  void i32(int32_t v) { raw(&v, sizeof(v)); }
  void u64(uint64_t v) { raw(&v, sizeof(v)); }
  void f32(float v) { raw(&v, sizeof(v)); }
  void bytes(const void* p, size_t n) { raw(p, n); }

  void str(const std::string& s) {
    u64((uint64_t)s.size());
    if (!s.empty()) raw(s.data(), s.size());
  }

  void pad_to(uint64_t alignment) {
    const uint64_t pos = (uint64_t)out_.tellp();
    const uint64_t rem = pos % alignment;
    if (rem == 0) return;
    static const char zeros[64] = {0};
    bytes(zeros, (size_t)(alignment - rem));
  }

  bool ok() const { return out_.good(); }

 private:
  void raw(const void* p, size_t n) {
    out_.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
  }
  std::ofstream out_;
};

void write_value(BinaryWriter& w, const GgufValue& v) {
  switch (v.kind) {
    case GgufValue::Kind::U32:
      w.u32(kU32);
      w.u32(v.u32);
      break;
    case GgufValue::Kind::I32:
      w.u32(kI32);
      w.i32(v.i32);
      break;
    case GgufValue::Kind::F32:
      w.u32(kF32);
      w.f32(v.f32);
      break;
    case GgufValue::Kind::Bool:
      w.u32(kBool);
      w.u8(v.b ? 1 : 0);
      break;
    case GgufValue::Kind::String:
      w.u32(kString);
      w.str(v.str);
      break;
    case GgufValue::Kind::StringArray:
      w.u32(kArray);
      w.u32(kString);
      w.u64((uint64_t)v.strings.size());
      for (const auto& s : v.strings) w.str(s);
      break;
    case GgufValue::Kind::F32Array:
      w.u32(kArray);
      w.u32(kF32);
      w.u64((uint64_t)v.f32s.size());
      for (float f : v.f32s) w.f32(f);
      break;
    case GgufValue::Kind::I32Array:
      w.u32(kArray);
      w.u32(kI32);
      w.u64((uint64_t)v.i32s.size());
      for (int32_t i : v.i32s) w.i32(i);
      break;
  }
}

// ---------------------------------------------------------------------------
// 状态字典键 → GGUF 张量名
// ---------------------------------------------------------------------------
bool parse_block_index(const std::string& key, int* index, std::string* rest) {
  // blocks.<i>.<rest>
  const std::string prefix = "blocks.";
  if (key.rfind(prefix, 0) != 0) return false;
  const size_t dot = key.find('.', prefix.size());
  if (dot == std::string::npos) return false;
  const std::string idx = key.substr(prefix.size(), dot - prefix.size());
  if (idx.empty()) return false;
  for (char c : idx) {
    if (c < '0' || c > '9') return false;
  }
  *index = std::stoi(idx);
  *rest = key.substr(dot + 1);
  return true;
}

// 返回空字符串表示跳过（如 mask buffer / 未识别键）
std::string map_tensor_name(const std::string& key) {
  if (key == "token_embedding.weight") return "token_embd.weight";
  if (key == "position_embedding.weight") return "position_embd.weight";
  if (key == "ln_f.weight") return "output_norm.weight";
  if (key == "ln_f.bias") return "output_norm.bias";
  if (key == "lm_head.weight") return "output.weight";
  // llama.cpp 的 gpt2 架构没有 output 偏置（GPT-2 的 lm_head 本身就是无偏置 + 权重共享）
  if (key == "lm_head.bias") return {};

  int index = 0;
  std::string rest;
  if (!parse_block_index(key, &index, &rest)) return {};
  const std::string blk = "blk." + std::to_string(index) + ".";
  if (rest == "ln1.weight") return blk + "attn_norm.weight";
  if (rest == "ln1.bias") return blk + "attn_norm.bias";
  if (rest == "attn.qkv.weight") return blk + "attn_qkv.weight";
  if (rest == "attn.qkv.bias") return blk + "attn_qkv.bias";
  if (rest == "attn.proj.weight") return blk + "attn_output.weight";
  if (rest == "attn.proj.bias") return blk + "attn_output.bias";
  if (rest == "ln2.weight") return blk + "ffn_norm.weight";
  if (rest == "ln2.bias") return blk + "ffn_norm.bias";
  if (rest == "ff.net.0.weight") return blk + "ffn_up.weight";
  if (rest == "ff.net.0.bias") return blk + "ffn_up.bias";
  if (rest == "ff.net.2.weight") return blk + "ffn_down.weight";
  if (rest == "ff.net.2.bias") return blk + "ffn_down.bias";
  return {}; // attn.mask 等 buffer 直接跳过
}

// ---------------------------------------------------------------------------
// SentencePiece 词表 → GGUF tokenizer 元数据
// ---------------------------------------------------------------------------
struct SpmVocab {
  std::vector<std::string> tokens;
  std::vector<float> scores;
  std::vector<int32_t> types;
  int32_t bos = -1;
  int32_t eos = -1;
  int32_t unk = -1;
};

bool load_spm_vocab(const fs::path& spm_path, SpmVocab* vocab, std::string* err) {
  sentencepiece::SentencePieceProcessor sp;
  const auto status = sp.Load(path_to_ansi(spm_path));
  if (!status.ok()) {
    if (err) *err = "加载分词器失败: " + status.ToString();
    return false;
  }
  const int n = sp.GetPieceSize();
  if (n <= 0) {
    if (err) *err = "分词器词表为空";
    return false;
  }
  vocab->tokens.reserve(n);
  vocab->scores.reserve(n);
  vocab->types.reserve(n);
  for (int i = 0; i < n; ++i) {
    vocab->tokens.push_back(sp.IdToPiece(i));
    vocab->scores.push_back(sp.GetScore(i));
    // GGUF token_type: 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 5=UNUSED 6=BYTE
    int32_t type = 1;
    if (sp.IsUnknown(i)) {
      type = 2;
    } else if (sp.IsControl(i)) {
      type = 3;
    } else if (sp.IsUnused(i)) {
      type = 5;
    } else if (sp.IsByte(i)) {
      type = 6;
    }
    vocab->types.push_back(type);
  }
  vocab->bos = sp.bos_id();
  vocab->eos = sp.eos_id();
  vocab->unk = sp.unk_id();
  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// 自动查找分词器
// ---------------------------------------------------------------------------
std::vector<fs::path> find_spm_candidates(const fs::path& model_dir) {
  std::vector<fs::path> candidates;
  if (!model_dir.empty()) {
    candidates.push_back(model_dir / "spm.model");
    candidates.push_back(model_dir.parent_path() / "spm.model");
  }
  const fs::path root = Config::project_root();
  candidates.push_back(root / "data" / "models" / "spm.model");
  candidates.push_back(root / "data" / "data" / "models" / "spm.model");
  candidates.push_back(root / "models" / "spm.model");
  candidates.push_back(root.parent_path() / u8path("py训练器") / "models" /
                       "spm.model");
  std::vector<fs::path> existing;
  for (const auto& c : candidates) {
    std::error_code ec;
    if (fs::is_regular_file(c, ec)) existing.push_back(c);
  }
  return existing;
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------
bool export_gpt_to_gguf(const fs::path& model_dir, const fs::path& out_path,
                        const fs::path& spm_model, GgufExportResult* result,
                        std::string* err) {
  auto fail = [err](const std::string& msg) {
    if (err) *err = msg;
    return false;
  };

  const fs::path config_path = model_dir / "config.json";
  const fs::path model_path = model_dir / "model.pt";
  std::error_code ec;
  if (!fs::is_regular_file(config_path, ec)) {
    return fail("模型目录缺少 config.json: " + path_to_utf8(model_dir));
  }
  if (!fs::is_regular_file(model_path, ec)) {
    return fail("模型目录缺少 model.pt: " + path_to_utf8(model_dir));
  }

  // ---- 读取超参 ----
  GPTConfig cfg;
  try {
    std::ifstream f(config_path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    const Json js = Json::parse(ss.str());
    cfg.vocab_size = js.get_int("vocab_size").value_or(8000);
    cfg.embed_dim = js.get_int("embed_dim").value_or(256);
    cfg.num_heads = js.get_int("num_heads").value_or(8);
    cfg.num_layers = js.get_int("num_layers").value_or(6);
    cfg.max_seq_len = js.get_int("max_seq_len").value_or(256);
  } catch (const std::exception& e) {
    return fail(std::string("解析 config.json 失败: ") + e.what());
  }

  // ---- 分词器（必须与模型词表大小一致，llama.cpp 要求 embedding 行数 = 词表数）----
  SpmVocab vocab;
  fs::path spm;
  std::string mismatch_note;
  if (!spm_model.empty()) {
    std::string load_err;
    SpmVocab v;
    if (!load_spm_vocab(spm_model, &v, &load_err)) return fail(load_err);
    if ((int64_t)v.tokens.size() != cfg.vocab_size) {
      return fail("分词器词表大小(" + std::to_string(v.tokens.size()) +
                  ") 与模型配置(" + std::to_string(cfg.vocab_size) +
                  ") 不一致，请选择训练该模型时使用的 spm.model");
    }
    spm = spm_model;
    vocab = std::move(v);
  } else {
    const auto candidates = find_spm_candidates(model_dir);
    for (const auto& cand : candidates) {
      SpmVocab v;
      std::string load_err;
      if (!load_spm_vocab(cand, &v, &load_err)) continue;
      if ((int64_t)v.tokens.size() == cfg.vocab_size) {
        spm = cand;
        vocab = std::move(v);
        break;
      }
      if (!mismatch_note.empty()) mismatch_note += "；";
      mismatch_note += path_to_utf8(cand) + "(" +
                       std::to_string(v.tokens.size()) + " 词表)";
    }
    if (spm.empty()) {
      std::string msg = "找不到与模型词表大小(" + std::to_string(cfg.vocab_size) +
                        ")匹配的 spm.model";
      if (!mismatch_note.empty()) msg += "；已找到: " + mismatch_note;
      return fail(msg);
    }
  }

  // ---- 读取权重 ----
  std::vector<std::pair<std::string, torch::Tensor>> sd;
  try {
    sd = load_legacy_state_dict(model_path, torch::kCPU);
  } catch (const std::exception& e) {
    return fail(std::string("读取 model.pt 失败: ") + e.what());
  }

  std::vector<GgufTensor> tensors;
  tensors.reserve(sd.size());
  for (auto& [key, tensor] : sd) {
    std::string name = map_tensor_name(key);
    if (name.empty()) continue; // mask buffer 等
    auto t = tensor.to(torch::kCPU).contiguous();
    if (t.scalar_type() != torch::kFloat32) t = t.to(torch::kFloat32);
    GgufTensor gt;
    gt.name = name;
    gt.type = kGgmlF32;
    // PyTorch [out, in] → GGUF ne = [in, out]
    std::vector<uint64_t> torch_dims;
    for (int64_t d : t.sizes()) torch_dims.push_back((uint64_t)d);
    gt.dims.assign(torch_dims.rbegin(), torch_dims.rend());
    if (gt.dims.empty()) gt.dims.push_back(1);
    gt.nbytes = (uint64_t)t.numel() * sizeof(float);
    gt.data = t;
    tensors.push_back(std::move(gt));
  }
  if (tensors.empty()) return fail("模型中没有可导出的权重");

  // 按名字排序，输出稳定
  std::sort(tensors.begin(), tensors.end(),
            [](const GgufTensor& a, const GgufTensor& b) { return a.name < b.name; });

  // ---- 组装元数据 ----
  std::vector<std::pair<std::string, GgufValue>> kvs;
  auto add_u32 = [&](const std::string& k, uint32_t v) {
    GgufValue x;
    x.kind = GgufValue::Kind::U32;
    x.u32 = v;
    kvs.emplace_back(k, x);
  };
  auto add_i32 = [&](const std::string& k, int32_t v) {
    GgufValue x;
    x.kind = GgufValue::Kind::I32;
    x.i32 = v;
    kvs.emplace_back(k, x);
  };
  auto add_f32 = [&](const std::string& k, float v) {
    GgufValue x;
    x.kind = GgufValue::Kind::F32;
    x.f32 = v;
    kvs.emplace_back(k, x);
  };
  auto add_bool = [&](const std::string& k, bool v) {
    GgufValue x;
    x.kind = GgufValue::Kind::Bool;
    x.b = v;
    kvs.emplace_back(k, x);
  };
  auto add_str = [&](const std::string& k, const std::string& v) {
    GgufValue x;
    x.kind = GgufValue::Kind::String;
    x.str = v;
    kvs.emplace_back(k, x);
  };

  add_str("general.architecture", "gpt2");
  add_str("general.name", path_to_utf8(model_dir.filename()));
  add_u32("general.file_type", 0); // ALL_F32
  add_u32("general.quantization_version", 2);
  add_u32("gpt2.context_length", (uint32_t)cfg.max_seq_len);
  add_u32("gpt2.embedding_length", (uint32_t)cfg.embed_dim);
  add_u32("gpt2.feed_forward_length", (uint32_t)(4 * cfg.embed_dim));
  add_u32("gpt2.block_count", (uint32_t)cfg.num_layers);
  add_u32("gpt2.attention.head_count", (uint32_t)cfg.num_heads);
  add_u32("gpt2.attention.head_count_kv", (uint32_t)cfg.num_heads);
  add_f32("gpt2.attention.layer_norm_epsilon", 1e-5f);

  add_str("tokenizer.ggml.model", "llama");
  add_str("tokenizer.ggml.pre", "default");
  {
    GgufValue x;
    x.kind = GgufValue::Kind::StringArray;
    x.strings = vocab.tokens;
    kvs.emplace_back("tokenizer.ggml.tokens", x);
  }
  {
    GgufValue x;
    x.kind = GgufValue::Kind::F32Array;
    x.f32s = vocab.scores;
    kvs.emplace_back("tokenizer.ggml.scores", x);
  }
  {
    GgufValue x;
    x.kind = GgufValue::Kind::I32Array;
    x.i32s = vocab.types;
    kvs.emplace_back("tokenizer.ggml.token_type", x);
  }
  if (vocab.bos >= 0) add_u32("tokenizer.ggml.bos_token_id", vocab.bos);
  if (vocab.eos >= 0) add_u32("tokenizer.ggml.eos_token_id", vocab.eos);
  if (vocab.unk >= 0) add_u32("tokenizer.ggml.unknown_token_id", vocab.unk);
  add_bool("tokenizer.ggml.add_bos_token", false);
  add_bool("tokenizer.ggml.add_eos_token", false);
  add_bool("tokenizer.ggml.add_space_prefix", true);

  // ---- 计算数据偏移 ----
  uint64_t offset = 0;
  for (auto& t : tensors) {
    t.offset = offset;
    offset += t.nbytes;
    const uint64_t rem = offset % kGgufAlignment;
    if (rem != 0) offset += kGgufAlignment - rem;
  }


  // ---- 写出 ----
  try {
    std::error_code ec2;
    if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path(), ec2);
    BinaryWriter w(out_path);

    w.u32(kGgufMagic);
    w.u32(kGgufVersion);
    w.u64((uint64_t)tensors.size());
    w.u64((uint64_t)kvs.size());
    for (const auto& [key, value] : kvs) {
      w.str(key);
      write_value(w, value);
    }
    for (const auto& t : tensors) {
      w.str(t.name);
      w.u32((uint32_t)t.dims.size());
      for (uint64_t d : t.dims) w.u64(d);
      w.u32(t.type);
      w.u64(t.offset);
    }
    w.pad_to(kGgufAlignment);
    for (const auto& t : tensors) {
      w.bytes(t.data.data_ptr<float>(), (size_t)t.nbytes);
      w.pad_to(kGgufAlignment);
    }
    if (!w.ok()) return fail("写入 GGUF 失败（磁盘错误？）");
  } catch (const std::exception& e) {
    return fail(std::string("写出 GGUF 失败: ") + e.what());
  }

  if (result) {
    result->out_path = out_path;
    result->spm_model = spm;
    result->tensor_count = (int64_t)tensors.size();
    result->vocab_size = (int64_t)vocab.tokens.size();
    std::error_code size_ec;
    result->bytes = (int64_t)fs::file_size(out_path, size_ec);
  }
  if (err) err->clear();
  return true;
}


// ---------------------------------------------------------------------------
// 读取 GGUF 词表大小（只解析 KV 段，不读张量）
// ---------------------------------------------------------------------------
namespace {

bool skip_gguf_value(std::ifstream& f, uint32_t type, uint64_t* array_count,
                     uint32_t* array_elem_type) {
  auto read = [&f](void* dst, size_t n) { f.read((char*)dst, (std::streamsize)n); return f.good(); };
  switch (type) {
    case kU8: case kI8: case kBool: { char c; return read(&c, 1); }
    case kU16: case kI16: { char c[2]; return read(c, 2); }
    case kU32: case kI32: case kF32: { char c[4]; return read(c, 4); }
    case kU64: case kI64: case kF64: { char c[8]; return read(c, 8); }
    case kString: {
      uint64_t n = 0;
      if (!read(&n, 8)) return false;
      f.seekg((std::streamoff)n, std::ios::cur);
      return f.good();
    }
    case kArray: {
      uint32_t et = 0;
      uint64_t count = 0;
      if (!read(&et, 4) || !read(&count, 8)) return false;
      if (array_count != nullptr) *array_count = count;
      if (array_elem_type != nullptr) *array_elem_type = et;
      if (et == kString) {
        for (uint64_t i = 0; i < count; ++i) {
          uint64_t n = 0;
          if (!read(&n, 8)) return false;
          f.seekg((std::streamoff)n, std::ios::cur);
          if (!f.good()) return false;
        }
        return true;
      }
      uint64_t elem = 0;
      switch (et) {
        case kU8: case kI8: case kBool: elem = 1; break;
        case kU16: case kI16: elem = 2; break;
        case kU32: case kI32: case kF32: elem = 4; break;
        case kU64: case kI64: case kF64: elem = 8; break;
        default: return false;
      }
      f.seekg((std::streamoff)(elem * count), std::ios::cur);
      return f.good();
    }
    default:
      return false;
  }
}

} // namespace

bool gguf_read_vocab_size(const fs::path& gguf_path, int64_t* vocab_size,
                          std::string* err) {
  auto fail = [err](const std::string& m) {
    if (err != nullptr) *err = m;
    return false;
  };
  std::ifstream f(gguf_path, std::ios::binary);
  if (!f) return fail("无法打开 GGUF: " + path_to_utf8(gguf_path));
  char magic[4] = {0};
  f.read(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) return fail("不是 GGUF 文件");
  uint32_t version = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  f.read((char*)&version, 4);
  f.read((char*)&n_tensors, 8);
  f.read((char*)&n_kv, 8);
  (void)n_tensors;
  for (uint64_t i = 0; i < n_kv; ++i) {
    uint64_t key_len = 0;
    if (!f.read((char*)&key_len, 8) || key_len > 65536) return fail("GGUF 头损坏");
    std::string key(key_len, '\0');
    f.read(key.data(), (std::streamsize)key_len);
    uint32_t type = 0;
    if (!f.read((char*)&type, 4)) return fail("GGUF 头损坏");
    uint64_t count = 0;
    uint32_t elem_type = 0;
    if (!skip_gguf_value(f, type, &count, &elem_type)) return fail("GGUF 头损坏");
    if (key == "tokenizer.ggml.tokens") {
      if (type != kArray || elem_type != kString) return fail("词表不是字符串数组");
      if (vocab_size != nullptr) *vocab_size = (int64_t)count;
      return true;
    }
  }
  return fail("GGUF 中没有 tokenizer.ggml.tokens");
}
} // namespace gpt
