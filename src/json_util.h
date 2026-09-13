#pragma once
// json_util.h - 极简 JSON 解析/序列化（读写 config.json 用）
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpt {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;

  Json() : type_(Type::Null) {}
  Json(bool b) : type_(Type::Bool), b_(b) {}
  Json(int n) : type_(Type::Number), n_((double)n) {}
  Json(int64_t n) : type_(Type::Number), n_((double)n) {}
  Json(double n) : type_(Type::Number), n_(n) {}
  Json(const char* s) : type_(Type::String), s_(s) {}
  Json(std::string s) : type_(Type::String), s_(std::move(s)) {}
  Json(Array a) : type_(Type::Array), a_(std::move(a)) {}
  Json(Object o) : type_(Type::Object), o_(std::move(o)) {}

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_object() const { return type_ == Type::Object; }
  bool is_array() const { return type_ == Type::Array; }

  bool as_bool() const { return b_; }
  double as_number() const { return n_; }
  int64_t as_int() const { return (int64_t)n_; }
  const std::string& as_string() const { return s_; }
  const Array& as_array() const { return a_; }
  const Object& as_object() const { return o_; }

  bool has(const std::string& key) const {
    return type_ == Type::Object && o_.count(key) != 0;
  }
  const Json& at(const std::string& key) const {
    if (type_ != Type::Object) throw std::runtime_error("JSON 节点不是对象: " + key);
    auto it = o_.find(key);
    if (it == o_.end()) throw std::runtime_error("JSON 缺少字段: " + key);
    return it->second;
  }

  std::optional<int64_t> get_int(const std::string& key) const;
  std::optional<double> get_number(const std::string& key) const;
  std::optional<std::string> get_string(const std::string& key) const;

  void set(const std::string& key, Json v);
  void push_back(Json v);

  // 序列化（indent < 0 表示紧凑）
  std::string dump(int indent = 2) const;

  // 解析，失败时抛 std::runtime_error
  static Json parse(const std::string& text);

 private:
  Type type_;
  bool b_ = false;
  double n_ = 0.0;
  std::string s_;
  Array a_;
  Object o_;
};

} // namespace gpt
