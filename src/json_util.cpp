#include "json_util.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace gpt {

// ---------------------------------------------------------------------------
// 取值辅助
// ---------------------------------------------------------------------------
std::optional<int64_t> Json::get_int(const std::string& key) const {
  if (!has(key)) return std::nullopt;
  const Json& v = at(key);
  if (v.type() == Type::Number) return (int64_t)v.n_;
  return std::nullopt;
}

std::optional<double> Json::get_number(const std::string& key) const {
  if (!has(key)) return std::nullopt;
  const Json& v = at(key);
  if (v.type() == Type::Number) return v.n_;
  return std::nullopt;
}

std::optional<std::string> Json::get_string(const std::string& key) const {
  if (!has(key)) return std::nullopt;
  const Json& v = at(key);
  if (v.type() == Type::String) return v.s_;
  return std::nullopt;
}

void Json::set(const std::string& key, Json v) {
  if (type_ != Type::Object) {
    type_ = Type::Object;
    o_.clear();
  }
  o_[key] = std::move(v);
}

void Json::push_back(Json v) {
  if (type_ != Type::Array) {
    type_ = Type::Array;
    a_.clear();
  }
  a_.push_back(std::move(v));
}

// ---------------------------------------------------------------------------
// 序列化
// ---------------------------------------------------------------------------
static void append_escaped(std::string& out, const std::string& s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back((char)c);
        }
    }
  }
  out.push_back('"');
}

static void dump_internal(const Json& j, std::string& out, int indent, int depth) {
  auto pad = [&](int d) {
    if (indent >= 0) out.append((size_t)(d * indent), ' ');
  };
  switch (j.type()) {
    case Json::Type::Null: out += "null"; break;
    case Json::Type::Bool: out += j.as_bool() ? "true" : "false"; break;
    case Json::Type::Number: {
      double n = j.as_number();
      if (std::isfinite(n) && n == (double)(int64_t)n) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", (long long)n);
        out += buf;
      } else {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", n);
        out += buf;
      }
      break;
    }
    case Json::Type::String: append_escaped(out, j.as_string()); break;
    case Json::Type::Array: {
      if (j.as_array().empty()) { out += "[]"; break; }
      out.push_back('[');
      bool first = true;
      for (const auto& item : j.as_array()) {
        if (!first) out.push_back(',');
        first = false;
        if (indent >= 0) { out.push_back('\n'); pad(depth + 1); }
        dump_internal(item, out, indent, depth + 1);
      }
      if (indent >= 0) { out.push_back('\n'); pad(depth); }
      out.push_back(']');
      break;
    }
    case Json::Type::Object: {
      if (j.as_object().empty()) { out += "{}"; break; }
      out.push_back('{');
      bool first = true;
      for (const auto& [k, v] : j.as_object()) {
        if (!first) out.push_back(',');
        first = false;
        if (indent >= 0) { out.push_back('\n'); pad(depth + 1); }
        append_escaped(out, k);
        out += ": ";
        dump_internal(v, out, indent, depth + 1);
      }
      if (indent >= 0) { out.push_back('\n'); pad(depth); }
      out.push_back('}');
      break;
    }
  }
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_internal(*this, out, indent, 0);
  return out;
}

// ---------------------------------------------------------------------------
// 解析
// ---------------------------------------------------------------------------
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Json parse() {
    Json v = parse_value();
    skip_ws();
    if (pos_ != text_.size()) fail("多余内容");
    return v;
  }

 private:
  const std::string& text_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("JSON 解析错误 @" + std::to_string(pos_) + ": " + msg);
  }

  void skip_ws() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r'))
      ++pos_;
  }

  char peek() {
    if (pos_ >= text_.size()) fail("意外结束");
    return text_[pos_];
  }

  void expect(char c) {
    if (pos_ >= text_.size() || text_[pos_] != c) fail(std::string("期望 '") + c + "'");
    ++pos_;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back((char)cp);
    } else if (cp < 0x800) {
      out.push_back((char)(0xC0 | (cp >> 6)));
      out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back((char)(0xE0 | (cp >> 12)));
      out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
      out.push_back((char)(0xF0 | (cp >> 18)));
      out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back((char)(0x80 | (cp & 0x3F)));
    }
  }

  static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) fail("字符串未闭合");
      char c = text_[pos_++];
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) fail("转义序列不完整");
      char e = text_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (pos_ + 4 > text_.size()) fail("\\u 转义不完整");
          uint32_t cp = 0;
          for (int i = 0; i < 4; ++i) {
            int h = hex_val(text_[pos_++]);
            if (h < 0) fail("非法 \\u 转义");
            cp = (cp << 4) | (uint32_t)h;
          }
          // 代理对
          if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= text_.size() &&
              text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
            pos_ += 2;
            uint32_t lo = 0;
            for (int i = 0; i < 4; ++i) {
              int h = hex_val(text_[pos_++]);
              if (h < 0) fail("非法 \\u 转义");
              lo = (lo << 4) | (uint32_t)h;
            }
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("非法转义字符");
      }
    }
    return out;
  }

  Json parse_number() {
    size_t start = pos_;
    if (peek() == '-') ++pos_;
    while (pos_ < text_.size() && isdigit((unsigned char)text_[pos_])) ++pos_;
    bool is_float = false;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      is_float = true;
      ++pos_;
      while (pos_ < text_.size() && isdigit((unsigned char)text_[pos_])) ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      is_float = true;
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && isdigit((unsigned char)text_[pos_])) ++pos_;
    }
    std::string num = text_.substr(start, pos_ - start);
    char* end = nullptr;
    double v = std::strtod(num.c_str(), &end);
    if (end == num.c_str() || *end != '\0') fail("非法数字: " + num);
    return Json(v);
  }

  Json parse_value() {
    skip_ws();
    char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Json(parse_string());
      case 't':
        if (text_.compare(pos_, 4, "true") == 0) { pos_ += 4; return Json(true); }
        fail("非法标识");
      case 'f':
        if (text_.compare(pos_, 5, "false") == 0) { pos_ += 5; return Json(false); }
        fail("非法标识");
      case 'n':
        if (text_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Json(); }
        fail("非法标识");
      default:
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(std::string("非法字符 '") + c + "'");
    }
  }

  Json parse_object() {
    expect('{');
    Json obj(Json::Object{});
    skip_ws();
    if (peek() == '}') { ++pos_; return obj; }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      obj.set(key, parse_value());
      skip_ws();
      char c = peek();
      if (c == ',') { ++pos_; continue; }
      if (c == '}') { ++pos_; break; }
      fail("对象格式错误");
    }
    return obj;
  }

  Json parse_array() {
    expect('[');
    Json arr(Json::Array{});
    skip_ws();
    if (peek() == ']') { ++pos_; return arr; }
    while (true) {
      arr.push_back(parse_value());
      skip_ws();
      char c = peek();
      if (c == ',') { ++pos_; continue; }
      if (c == ']') { ++pos_; break; }
      fail("数组格式错误");
    }
    return arr;
  }
};

} // namespace

Json Json::parse(const std::string& text) {
  Parser p(text);
  return p.parse();
}

} // namespace gpt
