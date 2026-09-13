#pragma once
// dataset.h - 数据集与 DataLoader（对应 python/dataset.py）
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace gpt {

// 读取 token_ids.txt（每行由空格分隔的 token id），返回完整 token 序列
std::vector<int64_t> load_token_ids(const std::filesystem::path& path,
                                    std::string* err = nullptr);

struct Batch {
  torch::Tensor x; // [B, seq]
  torch::Tensor y; // [B, seq]
};

// 滑动窗口数据集 + 每 epoch 洗牌、drop_last 的批加载器
class DataLoader {
 public:
  DataLoader(std::vector<int64_t> tokens, int64_t seq_len, int64_t batch_size,
             bool shuffle, bool drop_last, int64_t stride);

  int64_t num_samples() const { return (int64_t)starts_.size(); }
  int64_t num_batches() const {
    return batch_size_ > 0 ? (int64_t)order_.size() / batch_size_ : 0;
  }
  const std::vector<int64_t>& tokens() const { return tokens_; }

  void reshuffle();         // 每个 epoch 调用一次（对应 DataLoader shuffle=True）
  Batch batch(int64_t i) const; // 返回 CPU int64 张量

 private:
  std::vector<int64_t> tokens_;
  int64_t seq_len_ = 0;
  int64_t batch_size_ = 0;
  int64_t stride_ = 0;
  std::vector<int64_t> starts_;
  std::vector<int64_t> order_;
  mutable std::mt19937 rng_;
};

} // namespace gpt
