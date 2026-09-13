#pragma once
// path_util.h - Windows 路径编码转换（fs::path 宽字符 ↔ UTF-8/ANSI）
#include <filesystem>
#include <string>

namespace gpt {

// UTF-8 std::string → 宽字符串
std::wstring utf8_to_wide(const std::string& u8);
// 宽字符串 → UTF-8 std::string
std::string wide_to_utf8(const std::wstring& w);
// fs::path → UTF-8（用于显示与日志）
std::string path_to_utf8(const std::filesystem::path& p);
// fs::path → 本地 ANSI 编码（用于 sentencepiece 等 char* 路径 API）
std::string path_to_ansi(const std::filesystem::path& p);
// UTF-8 std::string → fs::path
std::filesystem::path u8path(const std::string& u8);

} // namespace gpt
