#include "dataset.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "path_util.h"

namespace gpt {

std::vector<int64_t> load_token_ids(const std::filesystem::path& path,
                                    std::string* err) {
  std::vector<int64_t> tokens;
  std::ifstream in(path);
  if (!in) {
    if (err) *err = "无法打开数据文件: " + path_to_utf8(path);
    return tokens;
  }
  std::string line;
  int64_t n = 0;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    std::string tok;
    bool any = false;
    while (iss >> tok) {
      char* end = nullptr;
      long long v = std::strtoll(tok.c_str(), &end, 10);
      if (end == tok.c_str() || *end != '\0') {
        // 跳过无法解析的 token（与 python int() 不同，容错处理）
        continue;
      }
      tokens.push_back((int64_t)v);
      any = true;
      ++n;
    }
    (void)any;
  }
  return tokens;
}

DataLoader::DataLoader(std::vector<int64_t> tokens, int64_t seq_len,
                       int64_t batch_size, bool shuffle, bool drop_last,
                       int64_t stride)
    : tokens_(std::move(tokens)),
      seq_len_(seq_len),
      batch_size_(batch_size),
      stride_(stride > 0 ? stride : seq_len),
      rng_(std::random_device{}()) {
  // 与 python 一致：range(0, len - seq_len, stride)，end = start + seq + 1 不越界
  int64_t n = (int64_t)tokens_.size() - seq_len_;
  if (n > 0) {
    int64_t count = (n + stride_ - 1) / stride_;
    starts_.reserve((size_t)count);
    for (int64_t s = 0; s < count; ++s) starts_.push_back(s * stride_);
  }
  if (tokens_.size() < (size_t)(seq_len_ + 1)) {
    printf("警告: 总 token 数 (%zu) 小于 seq_len+1 (%lld)，无法生成样本。\n",
           tokens_.size(), (long long)(seq_len_ + 1));
  }
  order_.resize(starts_.size());
  for (size_t i = 0; i < order_.size(); ++i) order_[i] = (int64_t)i;
  if (shuffle) reshuffle();
  (void)drop_last;
}

void DataLoader::reshuffle() {
  std::shuffle(order_.begin(), order_.end(), rng_);
}

Batch DataLoader::batch(int64_t i) const {
  const int64_t B = batch_size_;
  std::vector<int64_t> xbuf((size_t)(B * seq_len_));
  std::vector<int64_t> ybuf((size_t)(B * seq_len_));
  const int64_t base = i * B;
  for (int64_t b = 0; b < B; ++b) {
    const int64_t start = starts_[order_[(size_t)(base + b)]];
    const int64_t* src = tokens_.data() + start;
    std::copy(src, src + seq_len_, xbuf.data() + b * seq_len_);
    std::copy(src + 1, src + seq_len_ + 1, ybuf.data() + b * seq_len_);
  }
  Batch out;
  out.x = torch::tensor(xbuf, torch::TensorOptions().dtype(torch::kInt64))
              .reshape({B, seq_len_});
  out.y = torch::tensor(ybuf, torch::TensorOptions().dtype(torch::kInt64))
              .reshape({B, seq_len_});
  return out;
}

} // namespace gpt
