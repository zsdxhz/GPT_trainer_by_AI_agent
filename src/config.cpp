#include "config.h"

#include <torch/torch.h>

#include <windows.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
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
  // 1) 环境变量优先（宽字符读取，支持中文路径）
  std::wstring env(1024, L'\0');
  DWORD n = GetEnvironmentVariableW(L"GPT_ROOT", env.data(), (DWORD)env.size());
  if (n > 0 && n < env.size()) {
    env.resize(n);
    if (!env.empty()) return fs::path(env);
  }
  // 2) exe 同目录的 gpt_root.txt（发布包便携模式）：
  //    内容 "." 或空 = 根目录就是 exe 所在目录；否则按 UTF-8 路径解释
  {
    std::wstring buf(32768, L'\0');
    const DWORD len = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    if (len > 0 && len < buf.size()) {
      buf.resize(len);
      const fs::path exe_dir = fs::path(buf).parent_path();
      const fs::path marker = exe_dir / L"gpt_root.txt";
      std::error_code ec;
      if (fs::exists(marker, ec)) {
        std::ifstream f(marker, std::ios::binary);
        std::string line;
        std::getline(f, line);
        // 去掉 UTF-8 BOM / CR / 首尾空白
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
          line.erase(0, 3);
        }
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                                 line.back() == '\t')) {
          line.pop_back();
        }
        size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos) return exe_dir; // 空 = 便携模式
        line = line.substr(b);
        if (line == ".") return exe_dir;
        fs::path p = u8path(line);
        return p.is_absolute() ? p : (exe_dir / p);
      }
    }
  }
  // 3) 编译期根目录（开发机默认）
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
