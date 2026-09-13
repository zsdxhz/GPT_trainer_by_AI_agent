#include "config.h"

#include <torch/torch.h>

#include <windows.h>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

#include "json_util.h"
#include "path_util.h"
#include "zip_util.h"

// CMake 生成的根路径（build/gen/root_path.h）
#include "root_path.h"

#ifndef GPT_PROJECT_ROOT
#define GPT_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

namespace gpt {

fs::path Config::project_root() {
  // 环境变量优先（宽字符读取，支持中文路径）
  std::wstring env(1024, L'\0');
  DWORD n = GetEnvironmentVariableW(L"GPT_ROOT", env.data(), (DWORD)env.size());
  if (n > 0 && n < env.size()) {
    env.resize(n);
    if (!env.empty()) return fs::path(env);
  }
  return u8path(GPT_PROJECT_ROOT);
}

Config Config::defaults() {
  Config cfg;
  const fs::path root = project_root();
  cfg.model_save_dir = root / "models";
  cfg.log_dir = root / "logs";
  cfg.save_root = root / "saves";
  // 断点间隔环境变量覆盖（脚本化/测试用）
  if (const char* env = std::getenv("GPT_CHECKPOINT_INTERVAL")) {
    try {
      cfg.checkpoint_interval = std::stoll(env);
    } catch (...) {
    }
  }
  try {
    cfg.use_cuda = torch::cuda::is_available();
  } catch (...) {
    cfg.use_cuda = false;
  }
  return cfg;
}

void Config::init_dirs() const {
  std::error_code ec;
  fs::create_directories(model_save_dir, ec);
  fs::create_directories(log_dir, ec);
  fs::create_directories(save_root, ec);
}

// ---------------------------------------------------------------------------
// 数据集 zip 解压
// ---------------------------------------------------------------------------
static fs::path g_temp_dir; // 与 python 的全局 TEMP_DIR 对应

DatasetZipPaths load_dataset_from_zip(const fs::path& zip_path) {
  // 清理上次的临时目录
  if (!g_temp_dir.empty()) {
    std::error_code ec;
    fs::remove_all(g_temp_dir, ec);
    g_temp_dir.clear();
  }

  // 创建新的临时目录
  fs::path base = fs::temp_directory_path();
  for (int attempt = 0; attempt < 100; ++attempt) {
    std::ostringstream oss;
    oss << "gpt_dataset_" << std::hex << reinterpret_cast<uintptr_t>(&zip_path)
        << "_" << attempt;
    fs::path cand = base / utf8_to_wide(oss.str());
    std::error_code ec;
    if (fs::create_directory(cand, ec)) {
      g_temp_dir = cand;
      break;
    }
  }
  if (g_temp_dir.empty()) {
    throw std::runtime_error("无法创建临时目录");
  }

  std::string err;
  if (!extract_zip(zip_path, g_temp_dir, &err)) {
    throw std::runtime_error("解压 dataset.zip 失败: " + err);
  }

  DatasetZipPaths out;
  out.temp_dir = g_temp_dir;
  out.spm_model = g_temp_dir / "spm.model";
  out.token_ids = g_temp_dir / "token_ids.txt";
  out.config_json = g_temp_dir / "config.json";

  if (!fs::exists(out.spm_model))
    throw std::runtime_error("zip 包缺少 spm.model，路径: " + path_to_utf8(out.spm_model));
  if (!fs::exists(out.token_ids))
    throw std::runtime_error("zip 包缺少 token_ids.txt，路径: " + path_to_utf8(out.token_ids));
  if (!fs::exists(out.config_json))
    throw std::runtime_error("zip 包缺少 config.json，路径: " + path_to_utf8(out.config_json));

  return out;
}

} // namespace gpt
