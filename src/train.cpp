#include "train.h"

#include <ATen/autocast_mode.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "json_util.h"
#include "legacy_serial.h"
#include "path_util.h"

namespace fs = std::filesystem;

namespace gpt {

// ---------------------------------------------------------------------------
// CosineAnnealingLR（等价 python 实现）
// ---------------------------------------------------------------------------
CosineAnnealingLR::CosineAnnealingLR(torch::optim::Optimizer& optimizer,
                                     double T_max, double base_lr, double eta_min)
    : LRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min), base_lr_(base_lr) {}

std::vector<double> CosineAnnealingLR::get_lrs() {
  const double t = (double)step_count_; // step() 中先自增再取值，与 python 一致
  const double f = (1.0 + std::cos(M_PI * t / T_max_)) / 2.0;
  return {eta_min_ + (base_lr_ - eta_min_) * f};
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------
Trainer::Trainer(const Config& cfg, TrainerCallbacks callbacks,
                 const TrainerOptions& opts)
    : cfg_(cfg), cb_(std::move(callbacks)) {
  cfg_.init_dirs();

  // 设备
  if (cfg_.use_cuda && torch::cuda::is_available()) {
    device_ = torch::Device(torch::kCUDA);
  } else {
    device_ = torch::Device(torch::kCPU);
  }
  log("数据路径: " + path_to_utf8(cfg_.data_path));
  log(std::string("设备: ") + (device_.is_cuda() ? "cuda" : "cpu"));

  // 加载数据（滑动窗口）
  std::string err;
  auto tokens = load_token_ids(cfg_.data_path, &err);
  if (tokens.size() < (size_t)(cfg_.max_seq_len + 1)) {
    throw std::runtime_error(
        "数据为空！总 token 数不足 " + std::to_string(cfg_.max_seq_len + 1) +
        "，请检查 token_ids.txt 或减小 seq_len");
  }
  loader_ = std::make_unique<DataLoader>(tokens, cfg_.max_seq_len, cfg_.batch_size,
                                         /*shuffle=*/true, /*drop_last=*/true,
                                         cfg_.stride);
  total_samples = loader_->num_samples();
  steps_per_epoch = loader_->num_batches();
  log("总 token 数: " + std::to_string(tokens.size()));
  log("生成样本数: " + std::to_string(total_samples));
  log("总样本数: " + std::to_string(total_samples));
  log("每个 epoch 步数: " + std::to_string(steps_per_epoch));

  // 总训练步数
  if (cfg_.total_steps.has_value()) {
    total_steps = *cfg_.total_steps;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "将训练 %lld 步（约 %.1f 个 epoch）",
                  (long long)total_steps,
                  steps_per_epoch > 0 ? (double)total_steps / steps_per_epoch : 0.0);
    log(buf);
  } else {
    total_steps = steps_per_epoch * cfg_.num_epochs;
    log("将训练 " + std::to_string(cfg_.num_epochs) + " 个 epoch，总步数: " +
        std::to_string(total_steps));
  }

  // 模型
  GPTConfig mcfg;
  mcfg.vocab_size = cfg_.vocab_size;
  mcfg.embed_dim = cfg_.embed_dim;
  mcfg.num_heads = cfg_.num_heads;
  mcfg.num_layers = cfg_.num_layers;
  mcfg.max_seq_len = cfg_.max_seq_len;
  mcfg.dropout = cfg_.dropout;
  model = GPT(mcfg);
  model->to(device_);

  // 优化器 / 调度器
  torch::optim::AdamWOptions opt(cfg_.learning_rate);
  opt.weight_decay(cfg_.weight_decay);
  optimizer_ = std::make_unique<torch::optim::AdamW>(model->parameters(), opt);
  scheduler_ = std::make_unique<CosineAnnealingLR>(*optimizer_,
                                                   (double)total_steps,
                                                   cfg_.learning_rate, 1e-5);

  // TensorBoard 日志
  writer_ = std::make_unique<SummaryWriter>(cfg_.log_dir);

  // AMP：CUDA + bf16 可用时启用
  amp_enabled_ = device_.is_cuda();
  if (amp_enabled_) {
    try {
      auto t = torch::randn({2, 4},
                            torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
      torch::cuda::synchronize();
      (void)t;
    } catch (...) {
      amp_enabled_ = false;
    }
  }
  log(amp_enabled_ ? "混合精度: CUDA bf16 AMP (手动 GradScaler) 已启用"
                   : "混合精度: 未启用（CPU 或 bf16 不可用）");

  // 模型参数统计
  int64_t total_params = 0;
  for (const auto& p : model->parameters()) total_params += p.numel();
  log("模型参数量: " + std::to_string(total_params) + " (可训练: " +
      std::to_string(total_params) + ")");

  // ---- 初始权重（warm start）/ 断点恢复 ----
  if (!opts.init_model.empty() && opts.resume) {
    log("提示: 同时指定了初始权重与断点恢复，将优先从断点恢复");
  }
  if (!opts.init_model.empty() && !opts.resume) {
    log("从初始权重加载: " + path_to_utf8(opts.init_model));
    const auto sd = load_legacy_state_dict(opts.init_model, device_);
    load_state_dict_into(*model, sd);
    log("初始权重加载完成（优化器/学习率从新开始）");
  }
  if (opts.resume) {
    if (!load_training_state()) {
      log("未找到断点 (models/training_state/)，将从头开始训练");
    }
  }
}

Trainer::~Trainer() = default;

void Trainer::log(const std::string& line) {
  std::cout << line << std::endl;
  if (cb_.on_log) cb_.on_log(line);
}

// ---------------------------------------------------------------------------
// 训练一个 epoch
// ---------------------------------------------------------------------------
bool Trainer::train_epoch(long long max_steps) {
  model->train();
  double epoch_loss = 0.0;
  long long steps_in_epoch = 0;

  loader_->reshuffle();
  const int64_t n_batches = loader_->num_batches();

  for (int64_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    if (stop_training.load()) return false;
    if (max_steps >= 0 && steps_in_epoch >= max_steps) break;

    auto b = loader_->batch(batch_idx);
    auto x = b.x.to(device_);
    auto y = b.y.to(device_);

    optimizer_->zero_grad();

    // ---- 前向（AMP autocast）----
    if (amp_enabled_) {
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    }
    auto logits = model->forward(x);
    auto loss = torch::nn::functional::cross_entropy(
        logits.reshape({-1, cfg_.vocab_size}), y.reshape(-1));
    if (amp_enabled_) {
      at::autocast::set_autocast_enabled(at::kCUDA, false);
    }

    // ---- 反向 + 缩放 ----
    if (amp_enabled_) {
      (loss * grad_scale_).backward();
      // unscale 并检测 inf/nan
      bool found_inf = false;
      for (auto& p : model->parameters()) {
        auto g = p.grad();
        if (!g.defined()) continue;
        g.div_(grad_scale_);
        if (!torch::isfinite(g).all().item<bool>()) found_inf = true;
      }
      torch::nn::utils::clip_grad_norm_(model->parameters(), cfg_.grad_clip);
      if (!found_inf) {
        optimizer_->step();
        scheduler_->step();
      }
      // 更新缩放因子（每 2000 个正常步翻倍，上限 65536）
      if (found_inf) {
        grad_scale_ = std::max(grad_scale_ / 2.0, 1.0);
        good_steps_ = 0;
      } else if (++good_steps_ % 2000 == 0) {
        grad_scale_ = std::min(grad_scale_ * 2.0, 65536.0);
      }
    } else {
      loss.backward();
      torch::nn::utils::clip_grad_norm_(model->parameters(), cfg_.grad_clip);
      optimizer_->step();
      scheduler_->step();
    }

    const float loss_val = loss.item<float>();
    ++step;
    ++steps_in_epoch;
    epoch_loss += (double)loss_val;

    if (cb_.on_progress && step % 10 == 0) {
      current_loss = loss_val;
      cb_.on_progress(loss_val, current_epoch + 1, step, total_steps);
    }

    if (step % 20 == 0) {
      writer_->add_scalar("loss/train", loss_val, step);
      writer_->add_scalar("lr", (float)scheduler_->current_lr(), step);
    }

    // 周期性断点保存（含优化器状态，可完整续训）
    if (cfg_.checkpoint_interval > 0 && step % cfg_.checkpoint_interval == 0) {
      if (save_training_state()) {
        log("已保存断点 (step " + std::to_string(step) + "/" +
            std::to_string(total_steps) + ")");
      }
    }
  }

  // 平均 loss
  double avg_loss = steps_in_epoch > 0 ? epoch_loss / steps_in_epoch : 0.0;
  writer_->add_scalar("loss/epoch", (float)avg_loss, current_epoch + 1);

  // 保存检查点
  save_checkpoint(
      cfg_.model_save_dir /
          ("checkpoint_epoch_" + std::to_string(current_epoch + 1) + ".pt"),
      current_epoch + 1, avg_loss);

  ++current_epoch;
  return true;
}

// ---------------------------------------------------------------------------
// 完整训练
// ---------------------------------------------------------------------------
void Trainer::train() {
  log("开始训练，目标总步数: " + std::to_string(total_steps) +
      (step > 0 ? "（从断点继续）" : ""));
  const auto start_time = std::chrono::steady_clock::now();
  long long trained_steps = step; // 断点恢复时从已训步数继续

  try {
    while (trained_steps < total_steps) {
      const long long remaining = total_steps - trained_steps;
      const long long max_steps = std::min((long long)steps_per_epoch, remaining);
      const bool success = train_epoch(max_steps);
      trained_steps += max_steps;
      if (!success) {
        log("训练被用户停止。");
        break;
      }
      log("已完成 " + std::to_string(trained_steps) + " / " +
          std::to_string(total_steps) + " 步");
      if (trained_steps >= total_steps) break;
    }
  } catch (const std::exception& e) {
    log(std::string("训练出错: ") + e.what());
  }

  writer_->close();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
          .count();
  char buf[128];
  std::snprintf(buf, sizeof(buf), "训练结束，总耗时: %.2f 分钟", elapsed / 60.0);
  log(buf);

  // 保存最终模型（与 python torch.save(state_dict) 兼容，可互读）
  const fs::path final_path = cfg_.model_save_dir / "final_model.pt";
  save_legacy_state_dict(final_path, collect_state_dict(*model));
  log("最终模型已保存: " + path_to_utf8(final_path));

  // 断点处理：未完成 → 保留断点；已完成 → 清理断点目录
  if (step > 0 && step < total_steps) {
    try {
      if (save_training_state()) {
        log("断点已保存 (step " + std::to_string(step) + "/" +
            std::to_string(total_steps) + ")，下次勾选「从断点继续」即可续训");
      }
    } catch (const std::exception& e) {
      log(std::string("断点保存失败: ") + e.what());
    }
  } else if (step >= total_steps && total_steps > 0) {
    std::error_code ec;
    fs::remove_all(cfg_.model_save_dir / "training_state", ec);
  }

  // 自动保存到 saves/model{N}/
  save_model_to_saves();
}

// ---------------------------------------------------------------------------
// 检查点（近似 python torch.save({...}) 布局）
// ---------------------------------------------------------------------------
void Trainer::save_checkpoint(const fs::path& path, int64_t epoch,
                              double avg_loss) {
  torch::serialize::OutputArchive archive;
  archive.write("epoch", torch::tensor(epoch, torch::kInt64));
  archive.write("loss", torch::tensor(avg_loss, torch::kFloat64));

  torch::serialize::OutputArchive msd;
  model->save(msd);
  archive.write("model_state_dict", msd);

  // optimizer_state_dict（近似 python 结构：state 按参数序号命名）
  torch::serialize::OutputArchive osd;
  {
    torch::serialize::OutputArchive groups;
    torch::serialize::OutputArchive g0;
    g0.write("lr", torch::tensor(scheduler_->current_lr(), torch::kFloat64));
    g0.write("betas", torch::tensor({0.9, 0.999}, torch::kFloat64));
    g0.write("eps", torch::tensor(1e-8, torch::kFloat64));
    g0.write("weight_decay", torch::tensor(cfg_.weight_decay, torch::kFloat64));
    groups.write("0", g0);
    osd.write("param_groups", groups);
  }
  {
    torch::serialize::OutputArchive state;
    const auto& params = model->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
      auto it = optimizer_->state().find(params[i].unsafeGetTensorImpl());
      if (it == optimizer_->state().end()) continue;
      const auto& st = static_cast<torch::optim::AdamWParamState&>(*it->second);
      torch::serialize::OutputArchive si;
      si.write("step", torch::tensor(st.step(), torch::kInt64));
      si.write("exp_avg", st.exp_avg());
      si.write("exp_avg_sq", st.exp_avg_sq());
      state.write(std::to_string(i), si);
    }
    osd.write("state", state);
  }
  archive.write("optimizer_state_dict", osd);

  archive.save_to(path.string());
}

// ---------------------------------------------------------------------------
// 保存到 saves/model{N}/
// ---------------------------------------------------------------------------
std::string Trainer::save_model_to_saves() {
  std::vector<int> numbers;
  std::error_code ec;
  if (fs::exists(cfg_.save_root)) {
    for (const auto& entry : fs::directory_iterator(cfg_.save_root, ec)) {
      if (!entry.is_directory()) continue;
      const std::string name = entry.path().filename().string();
      if (name.rfind("model", 0) == 0) {
        const std::string suffix = name.substr(5);
        if (!suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
          numbers.push_back(std::stoi(suffix));
        }
      }
    }
  }
  const int next_num = numbers.empty() ? 1 : *std::max_element(numbers.begin(), numbers.end()) + 1;
  const fs::path save_dir =
      cfg_.save_root / ("model" + std::to_string(next_num));
  fs::create_directories(save_dir);

  // 裸 state_dict（与 python torch.save(state_dict) 一致，可互读）
  save_legacy_state_dict(save_dir / "model.pt", collect_state_dict(*model));

  // config.json
  Json::Object obj;
  obj["vocab_size"] = Json(cfg_.vocab_size);
  obj["embed_dim"] = Json(cfg_.embed_dim);
  obj["num_heads"] = Json(cfg_.num_heads);
  obj["num_layers"] = Json(cfg_.num_layers);
  obj["max_seq_len"] = Json(cfg_.max_seq_len);
  obj["dropout"] = Json(cfg_.dropout);
  Json cfg_json(std::move(obj));
  {
    std::ofstream f(save_dir / "config.json");
    f << cfg_json.dump(2);
  }

  // info.txt
  {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char timebuf[64];
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm);

    std::ofstream f(save_dir / "info.txt");
    f << "Trained on " << timebuf << "\n";
    f << "Data: " << path_to_utf8(cfg_.data_path) << "\n";
    f << "Total Steps: " << total_steps << "\n";
    char lossbuf[64];
    std::snprintf(lossbuf, sizeof(lossbuf), "Final Loss: %.4f\n", (double)current_loss);
    f << lossbuf;
  }

  log("模型已保存到: " + path_to_utf8(save_dir));
  return path_to_utf8(save_dir);
}

// ---------------------------------------------------------------------------
// 断点：完整训练状态
//   models/training_state/model.pt  模型权重（legacy 格式）
//   models/training_state/optim.pt  优化器状态（step/exp_avg/exp_avg_sq 张量）
//   models/training_state/meta.json 步数/epoch/总步数/学习率/AMP 缩放因子
// ---------------------------------------------------------------------------
bool Trainer::save_training_state() {
  const fs::path dir = cfg_.model_save_dir / "training_state";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) return false;

  // 模型权重
  save_legacy_state_dict(dir / "model.pt", collect_state_dict(*model));

  // 优化器状态（按参数序号命名）
  std::vector<std::pair<std::string, torch::Tensor>> opt;
  const auto& params = model->parameters();
  for (size_t i = 0; i < params.size(); ++i) {
    auto it = optimizer_->state().find(params[i].unsafeGetTensorImpl());
    if (it == optimizer_->state().end()) continue; // 尚未初始化的参数
    const auto& st = static_cast<torch::optim::AdamWParamState&>(*it->second);
    opt.emplace_back("step_" + std::to_string(i),
                     torch::tensor((int64_t)st.step(), torch::kInt64).to(device_));
    opt.emplace_back("exp_avg_" + std::to_string(i), st.exp_avg());
    opt.emplace_back("exp_avg_sq_" + std::to_string(i), st.exp_avg_sq());
  }
  save_legacy_state_dict(dir / "optim.pt", opt);

  // 元信息
  Json::Object obj;
  obj["step"] = Json(step);
  obj["current_epoch"] = Json(current_epoch);
  obj["total_steps"] = Json(total_steps);
  obj["lr"] = Json(cfg_.learning_rate);
  obj["grad_scale"] = Json(grad_scale_);
  obj["good_steps"] = Json(good_steps_);
  obj["checkpoint_interval"] = Json(cfg_.checkpoint_interval);
  std::ofstream f(dir / "meta.json");
  f << Json(std::move(obj)).dump(2);
  return true;
}

bool Trainer::load_training_state() {
  const fs::path dir = cfg_.model_save_dir / "training_state";
  if (!fs::exists(dir / "model.pt") || !fs::exists(dir / "optim.pt") ||
      !fs::exists(dir / "meta.json")) {
    return false;
  }

  // 模型权重
  const auto sd = load_legacy_state_dict(dir / "model.pt", device_);
  load_state_dict_into(*model, sd);

  // 优化器状态
  const auto omap = load_legacy_state_dict(dir / "optim.pt", device_);
  std::map<std::string, torch::Tensor> om;
  for (const auto& [k, t] : omap) om[k] = t;

  const auto& params = model->parameters();
  for (size_t i = 0; i < params.size(); ++i) {
    auto it = optimizer_->state().find(params[i].unsafeGetTensorImpl());
    torch::optim::AdamWParamState* st = nullptr;
    if (it == optimizer_->state().end()) {
      auto ns = std::make_unique<torch::optim::AdamWParamState>();
      st = ns.get();
      optimizer_->state().emplace(params[i].unsafeGetTensorImpl(),
                                  std::move(ns));
    } else {
      st = static_cast<torch::optim::AdamWParamState*>(it->second.get());
    }
    auto s1 = om.find("step_" + std::to_string(i));
    auto s2 = om.find("exp_avg_" + std::to_string(i));
    auto s3 = om.find("exp_avg_sq_" + std::to_string(i));
    if (s1 != om.end()) st->step(s1->second.item<int64_t>());
    if (s2 != om.end()) st->exp_avg(s2->second);
    if (s3 != om.end()) st->exp_avg_sq(s3->second);
  }

  // 元信息
  std::ifstream mf(dir / "meta.json", std::ios::binary);
  std::ostringstream mss;
  mss << mf.rdbuf();
  Json meta = Json::parse(mss.str());
  step = meta.get_int("step").value_or(0);
  current_epoch = (int)meta.get_int("current_epoch").value_or(0);
  total_steps = meta.get_int("total_steps").value_or(total_steps);
  grad_scale_ = meta.get_number("grad_scale").value_or(grad_scale_);
  good_steps_ = (int64_t)meta.get_int("good_steps").value_or(0);
  const double saved_lr = meta.get_number("lr").value_or(cfg_.learning_rate);

  // 恢复优化器学习率与调度器位置（余弦退火曲线接续）
  static_cast<torch::optim::AdamWOptions&>(
      optimizer_->param_groups()[0].options())
      .lr(saved_lr);
  scheduler_->set_base_lr(saved_lr);
  scheduler_->set_step_count(step);

  log("已从断点恢复: step " + std::to_string(step) + "/" +
      std::to_string(total_steps) + ", epoch " + std::to_string(current_epoch) +
      ", lr " + std::to_string(saved_lr));
  return true;
}

} // namespace gpt
