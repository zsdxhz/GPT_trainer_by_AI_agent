#include "legacy_serial.h"

#include <torch/csrc/jit/serialization/unpickler.h>
#include <caffe2/serialize/inline_container.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "path_util.h"

namespace gpt {

namespace {

// ---------------------------------------------------------------------------
// pickle 字节修补
//
// python torch.save(OrderedDict 类型的 state_dict) 会通过
//   GLOBAL collections.OrderedDict + EMPTY_TUPLE + REDUCE  构造顶层容器，
// 而 libtorch 的 Unpickler 对该 GLOBAL 的特例（backward-hooks 场景）会把
// 栈顶替换为 None，导致后续 SETITEMS 崩溃。
// 这里做两处等价的字节级改写：
//   1. 顶层 "GLOBAL collections.OrderedDict BINPUT0 EMPTY_TUPLE REDUCE"
//      → "EMPTY_DICT"（普通字典，语义等价）
//   2. 每个张量的 backward-hooks 构造 "BINGET0 EMPTY_TUPLE REDUCE"
//      → "NEWFALSE"（该参数在反序列化时被跳过，值无意义）
// 未被改写的字节原样保留（memo 编号不受影响）。
// ---------------------------------------------------------------------------
constexpr char kTopPrefix[] = "ccollections\nOrderedDict\nq\x00)R"; // 29 字节
constexpr size_t kTopLen = sizeof(kTopPrefix) - 1;
constexpr char kHooksPattern[] = "h\x00)R";                        // 4 字节
constexpr size_t kHooksLen = sizeof(kHooksPattern) - 1;

std::string patch_state_dict_pickle(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  size_t i = 0;
  // 顶层前缀：位置在 "PROTO 2"（\x80\x02）之后（用 memcmp，compare 会因内嵌 \0 截断）
  if (in.size() >= 2 + kTopLen &&
      std::memcmp(in.data() + 2, kTopPrefix, kTopLen) == 0) {
    out.append(in, 0, 2);
    out.push_back('}'); // EMPTY_DICT
    i = 2 + kTopLen;
    // 其后的 "q\x01"（BINPUT 1）与 "("（MARK）原样保留
  }
  for (; i < in.size(); ++i) {
    if (i + kHooksLen <= in.size() &&
        std::memcmp(in.data() + i, kHooksPattern, kHooksLen) == 0) {
      out.push_back(static_cast<char>(0x89)); // NEWFALSE
      i += kHooksLen - 1;                     // 循环 ++i 后共前进 kHooksLen 字节
    } else {
      out.push_back(in[i]);
    }
  }
  return out;
}

} // namespace

// ---------------------------------------------------------------------------
// 读取
// ---------------------------------------------------------------------------
std::vector<std::pair<std::string, torch::Tensor>> load_legacy_state_dict(
    const std::filesystem::path& path, c10::optional<torch::Device> device) {
  caffe2::serialize::PyTorchStreamReader reader(path_to_ansi(path));

  // 不同 torch 版本的记录前缀不同：
  //   torch>=2.14 直接保存 → "<文件名主干>/data.pkl"
  //   旧版 torch → "archive/data.pkl"，更老的版本 → "data.pkl"（无前缀）
  std::string pickle_prefix;
  std::string tensor_prefix;
  std::vector<std::string> candidates = {
      path.stem().string() + "/data.pkl", "archive/data.pkl", "data.pkl",
      "model/data.pkl"};
  for (const auto& cand : candidates) {
    if (reader.hasRecord(cand)) {
      pickle_prefix = cand.substr(0, cand.size() - 4); // 去掉 ".pkl"
      tensor_prefix = pickle_prefix + "/";
      break;
    }
  }
  if (pickle_prefix.empty()) {
    throw std::runtime_error("不是 PyTorch state_dict 文件（缺少 data.pkl）: " +
                             path_to_utf8(path));
  }

  // 读取并修补 pickle
  auto [pickle_ptr, pickle_size] = reader.getRecord(pickle_prefix + ".pkl");
  std::string pickle((const char*)pickle_ptr.get(), pickle_size);
  pickle = patch_state_dict_pickle(pickle);

  size_t bytes_read = 0;
  auto read_fn = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle.size()) return 0;
    len = std::min(pickle.size() - bytes_read, len);
    std::memcpy(buffer, pickle.data() + bytes_read, len);
    bytes_read += len;
    return len;
  };
  auto read_record = [&](const std::string& name) -> at::DataPtr {
    return std::get<0>(reader.getRecord(tensor_prefix + name));
  };

  torch::jit::Unpickler unpickler(read_fn, /*type_resolver=*/nullptr,
                                  /*obj_loader=*/nullptr, read_record, device,
                                  /*use_storage_device=*/false);
  unpickler.set_version(reader.version());
  torch::IValue ivalue;
  try {
    ivalue = unpickler.parse_ivalue();
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << e.what() << " [pickle 解析失败于字节偏移 " << bytes_read << "/"
        << pickle.size() << "]";
    throw std::runtime_error(oss.str());
  }

  if (!ivalue.isGenericDict()) {
    throw std::runtime_error("state_dict 顶层不是字典: " + path_to_utf8(path));
  }

  std::vector<std::pair<std::string, torch::Tensor>> out;
  const auto dict = ivalue.toGenericDict();
  for (const auto& item : dict) {
    if (item.value().isTensor()) {
      out.emplace_back(item.key().toStringRef(), item.value().toTensor());
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 写出：pickle（协议 2）构造
// ---------------------------------------------------------------------------
namespace {

void put_bytes(std::string& out, const void* data, size_t n) {
  out.append((const char*)data, n);
}

void put_binunicode(std::string& out, const std::string& s) {
  out.push_back('X'); // BINUNICODE: uint32 LE 长度 + UTF-8
  uint32_t len = (uint32_t)s.size();
  for (int i = 0; i < 4; ++i) out.push_back((char)((len >> (8 * i)) & 0xFF));
  out += s;
}

void put_binint(std::string& out, int64_t v) {
  if (v >= 0 && v < 256) {
    out.push_back('K'); // BININT1（1 字节无符号）
    out.push_back((char)v);
  } else {
    out.push_back('J'); // BININT（4 字节 LE 有符号）
    uint32_t u = (uint32_t)(int32_t)v;
    for (int i = 0; i < 4; ++i) out.push_back((char)((u >> (8 * i)) & 0xFF));
  }
}

void put_global(std::string& out, const std::string& module,
                const std::string& name) {
  out.push_back('c'); // GLOBAL
  out += module;
  out.push_back('\n');
  out += name;
  out.push_back('\n');
}

// torch.<Type>Storage 全局名（python 持久化 tensor 使用的 dtype 表示）
std::string storage_global_name(torch::ScalarType t) {
  switch (t) {
    case torch::kFloat32: return "FloatStorage";
    case torch::kFloat64: return "DoubleStorage";
    case torch::kInt64: return "LongStorage";
    case torch::kInt32: return "IntStorage";
    case torch::kInt16: return "ShortStorage";
    case torch::kInt8: return "CharStorage";
    case torch::kUInt8: return "ByteStorage";
    case torch::kBool: return "BoolStorage";
    case torch::kFloat16: return "HalfStorage";
    case torch::kBFloat16: return "BFloat16Storage";
    default:
      throw std::runtime_error("state_dict 写出不支持该 dtype");
  }
}

// 单个 tensor 的 pickle 值：
//   torch._utils._rebuild_tensor_v2(BINPERSID(("storage", torch.XStorage, key,
//     "cpu", numel)), 0, size, stride, False, OrderedDict())
void put_tensor_value(std::string& out, const torch::Tensor& t, int storage_key) {
  put_global(out, "torch._utils", "_rebuild_tensor_v2");
  out.push_back('('); // MARK（参数列表）
  out.push_back('('); // MARK（storage 五元组）
  put_binunicode(out, "storage");
  put_global(out, "torch", storage_global_name(t.scalar_type()));
  put_binunicode(out, std::to_string(storage_key));
  put_binunicode(out, "cpu");
  put_binint(out, t.numel());
  out.push_back('t'); // TUPLE
  out.push_back('Q'); // BINPERSID
  put_binint(out, 0); // storage_offset
  out.push_back('('); // MARK（size）
  for (int64_t d : t.sizes()) put_binint(out, d);
  out.push_back('t'); // TUPLE
  out.push_back('('); // MARK（stride）
  for (int64_t s : t.strides()) put_binint(out, s);
  out.push_back('t'); // TUPLE
  out.push_back(static_cast<char>(0x89)); // NEWFALSE（requires_grad=False）
  put_global(out, "collections", "OrderedDict");
  out.push_back(')'); // EMPTY_TUPLE
  out.push_back('R'); // REDUCE → OrderedDict()
  out.push_back('t'); // TUPLE（6 参数）
  out.push_back('R'); // REDUCE → _rebuild_tensor_v2(...)
}

std::string build_state_dict_pickle(
    const std::vector<std::pair<std::string, torch::Tensor>>& tensors) {
  std::string p;
  p += "\x80\x02";        // PROTO 2
  put_bytes(p, "\x7d\x71\x00", 3); // EMPTY_DICT BINPUT 0（含 \0，必须按字节写入）
  p += "\x28";            // MARK
  int key = 0;
  for (const auto& [name, t] : tensors) {
    put_binunicode(p, name);
    put_tensor_value(p, t, key);
    ++key;
  }
  p += "\x75\x2e"; // SETITEMS STOP
  return p;
}

} // namespace

void save_legacy_state_dict(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, torch::Tensor>>& tensors) {
  // 注意：第一个参数是输出文件名（记录前缀按文件名主干生成，如 model.pt →
  // "model/data.pkl"），与 python torch.save(path) 的布局一致
  caffe2::serialize::PyTorchStreamWriter writer(path_to_ansi(path));

  const std::string pickle = build_state_dict_pickle(tensors);
  writer.writeRecord("data.pkl", pickle.data(), pickle.size());
  writer.writeRecord("byteorder", "little", 6);

  int key = 0;
  for (const auto& [name, t] : tensors) {
    (void)name;
    auto cpu = t.to(torch::kCPU).contiguous();
    writer.writeRecord("data/" + std::to_string(key), cpu.data_ptr(),
                       (size_t)cpu.nbytes());
    ++key;
  }
  writer.writeEndOfFile();
}

// ---------------------------------------------------------------------------
// 模型 state_dict 收集 / 严格加载
// ---------------------------------------------------------------------------
std::vector<std::pair<std::string, torch::Tensor>> collect_state_dict(
    const torch::nn::Module& module) {
  std::vector<std::pair<std::string, torch::Tensor>> out;
  for (const auto& item : module.named_parameters())
    out.emplace_back(item.key(), item.value());
  for (const auto& item : module.named_buffers())
    out.emplace_back(item.key(), item.value());
  return out;
}

void load_state_dict_into(
    torch::nn::Module& module,
    const std::vector<std::pair<std::string, torch::Tensor>>& sd) {
  auto named = module.named_parameters();
  auto buffers = module.named_buffers();

  std::map<std::string, torch::Tensor> src;
  for (const auto& [k, t] : sd) src[k] = t;

  // 与 python load_state_dict 一致：在 no_grad 下复制（叶张量 requires_grad 不允许普通 in-place）
  torch::NoGradGuard guard;
  std::vector<std::string> missing;
  for (auto& item : named) {
    auto it = src.find(item.key());
    if (it == src.end()) {
      missing.push_back(item.key());
    } else {
      item.value().copy_(it->second);
    }
  }
  for (auto& item : buffers) {
    auto it = src.find(item.key());
    if (it == src.end()) {
      missing.push_back(item.key());
    } else {
      item.value().copy_(it->second);
    }
  }

  std::vector<std::string> unexpected;
  for (const auto& [k, t] : sd) {
    if (!named.contains(k) && !buffers.contains(k)) unexpected.push_back(k);
  }

  if (!missing.empty() || !unexpected.empty()) {
    std::ostringstream oss;
    if (!missing.empty()) {
      oss << "缺少键: ";
      for (size_t i = 0; i < missing.size(); ++i)
        oss << (i ? ", " : "") << missing[i];
    }
    if (!unexpected.empty()) {
      if (!missing.empty()) oss << "; ";
      oss << "多余键: ";
      for (size_t i = 0; i < unexpected.size(); ++i)
        oss << (i ? ", " : "") << unexpected[i];
    }
    throw std::runtime_error("state_dict 键不匹配 — " + oss.str());
  }
}

} // namespace gpt
