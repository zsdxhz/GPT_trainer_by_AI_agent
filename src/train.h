#pragma once
// train.h - 训练核心逻辑（对应 python/train.py）
#include <torch/torch.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "config.h"
#include "dataset.h"
#include "model.h"
#include "tfevents.h"

namespace gpt {

// 与 python torch.optim.lr_scheduler.CosineAnnealingLR 等价的实现
// （libtorch 2.14 的 C++ API 未暴露该类）
class CosineAnnealingLR : public torch::optim::LRScheduler {
 public:
  CosineAnnealingLR(torch::optim::Optimizer& optimizer, double T_max,
                    double base_lr, double eta_min = 0.0);
  // 当前学习率（对应 python scheduler.get_last_lr()[0]）
  double current_lr() {
    const auto lrs = get_current_lrs();
    return lrs.empty() ? 0.0 : lrs[0];
  }
  // 断点恢复用
  long long step_count() const { return step_count_; }
  void set_step_count(long long n) { step_count_ = n; }
  void set_base_lr(double lr) { base_lr_ = lr; }

 protected:
  std::vector<double> get_lrs() override;

 private:
  double T_max_;
  double eta_min_;
  double base_lr_;
};

struct TrainerCallbacks {
  // 每 10 步回调一次（loss, epoch, step, total_steps），由工作线程调用
  std::function<void(float loss, int epoch, long long step, long long total_steps)>
      on_progress;
  // 日志回调（工作线程调用）
  std::function<void(const std::string& line)> on_log;
};

struct TrainerOptions {
  // 从断点恢复（models/training_state/）
  bool resume = false;
  // 从已有模型权重初始化（warm start，优化器/调度器全新开始）
  std::filesystem::path init_model;
};

class Trainer {
 public:
  Trainer(const Config& cfg, TrainerCallbacks callbacks = {},
          const TrainerOptions& opts = {});
  ~Trainer();

  // 完整训练流程（按 TOTAL_STEPS / NUM_EPOCHS 控制）
  void train();
  // 训练一个 epoch，最多 max_steps 步；正常结束返回 true，被停止返回 false
  bool train_epoch(long long max_steps);
  // 保存到 saves/model{N}/（自动编号）
  std::string save_model_to_saves();
  // 断点：完整训练状态（模型+优化器+调度器+步数+AMP 缩放因子）
  bool save_training_state();
  bool load_training_state();

  std::atomic<bool> stop_training{false};
  float current_loss = 0.0f;
  int current_epoch = 0;
  long long step = 0;
  long long total_steps = 0;
  int64_t steps_per_epoch = 0;
  int64_t total_samples = 0;
  GPT model{nullptr};
  const Config& cfg() const { return cfg_; }

 private:
  void log(const std::string& line);
  void save_checkpoint(const std::filesystem::path& path, int64_t epoch,
                       double avg_loss);

  Config cfg_;
  TrainerCallbacks cb_;
  torch::Device device_{torch::kCPU};
  std::unique_ptr<DataLoader> loader_;
  std::unique_ptr<torch::optim::AdamW> optimizer_;
  std::unique_ptr<CosineAnnealingLR> scheduler_;
  std::unique_ptr<SummaryWriter> writer_;

  // ---- AMP（bf16 混合精度，手动 GradScaler）----
  bool amp_enabled_ = false;
  double grad_scale_ = 65536.0;
  int64_t good_steps_ = 0;
};

} // namespace gpt
