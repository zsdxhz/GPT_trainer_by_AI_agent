#pragma once
// config.h - 训练配置（与 py训练器/python/config.py 对应）
// 路径字段统一使用 fs::path（Windows 下宽字符），避免中文路径编码问题。
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace gpt {

// 数据集 zip 解压结果
struct DatasetZipPaths {
  std::filesystem::path spm_model;   // 解压后的 spm.model 路径
  std::filesystem::path token_ids;   // 解压后的 token_ids.txt 路径
  std::filesystem::path config_json; // 解压后的 config.json 路径
  std::filesystem::path temp_dir;    // 解压目标目录（下次加载前会被清理）
};

class Config {
 public:
  // ---------- 路径 ----------
  std::filesystem::path data_path = std::filesystem::path(L"E:/data/processed/token_ids.txt");
  std::filesystem::path model_save_dir; // <root>/models
  std::filesystem::path log_dir;        // <root>/logs
  std::filesystem::path save_root;      // <root>/saves

  // ---------- 模型参数 ----------
  int64_t vocab_size = 8000;
  int64_t embed_dim = 256;
  int64_t num_heads = 8;
  int64_t num_layers = 6;
  int64_t max_seq_len = 256;
  double dropout = 0.1;

  // ---------- 训练参数 ----------
  int64_t batch_size = 16;
  double learning_rate = 3e-4;
  int64_t num_epochs = 10; // 仅在 total_steps 为空时生效
  double weight_decay = 0.01;
  double grad_clip = 1.0;

  // ---------- 滑动窗口步长 ----------
  int64_t stride = 128;

  // ---------- 训练总步数（空 = 按 epoch 数训练） ----------
  std::optional<long long> total_steps;

  // ---------- 断点保存间隔（步，0 = 不保存；可用环境变量 GPT_CHECKPOINT_INTERVAL 覆盖） ----------
  int64_t checkpoint_interval = 5000;

  // ---------- 设备 ----------
  bool use_cuda = false; // 构造 defaults() 时自动探测

  // 使用默认值并基于项目根目录初始化所有目录
  static Config defaults();
  // 创建 models/logs/saves 目录
  void init_dirs() const;
  // 项目根目录（exe 编译期写入，可用环境变量 GPT_ROOT 覆盖）
  static std::filesystem::path project_root();
};

// 解压 dataset.zip，校验并返回内部文件路径（对应 python 的 load_dataset_from_zip）
DatasetZipPaths load_dataset_from_zip(const std::filesystem::path& zip_path);

} // namespace gpt
