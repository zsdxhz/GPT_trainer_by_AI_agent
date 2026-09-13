#pragma once
// zip_util.h - 极简 ZIP 解压/打包（支持 Python zipfile 兼容的 deflate + store）
#include <filesystem>
#include <string>
#include <vector>

namespace gpt {

// 将 zip_path 中所有文件解压到 out_dir（自动创建子目录）
// 成功返回 true；失败返回 false 并在 err 中写入原因
bool extract_zip(const std::filesystem::path& zip_path,
                 const std::filesystem::path& out_dir, std::string* err);

// 打包：entries = (zip 内文件名, 本地文件路径)，使用 deflate 压缩，
// 带 UTF-8 文件名标志，Python zipfile 与本项目的 extract_zip 均可读取。
bool create_zip(const std::filesystem::path& zip_path,
                const std::vector<std::pair<std::string, std::filesystem::path>>& entries,
                std::string* err);

} // namespace gpt
