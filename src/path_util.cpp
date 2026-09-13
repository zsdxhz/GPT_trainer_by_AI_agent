#include "path_util.h"

#include <windows.h>

namespace fs = std::filesystem;

namespace gpt {

std::wstring utf8_to_wide(const std::string& u8) {
  if (u8.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, u8.data(), (int)u8.size(),
                                    nullptr, 0);
  std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
  if (n > 0) {
    MultiByteToWideChar(CP_UTF8, 0, u8.data(), (int)u8.size(), w.data(), n);
  }
  return w;
}

std::string wide_to_utf8(const std::wstring& w) {
  if (w.empty()) return std::string();
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr,
                                    0, nullptr, nullptr);
  std::string s((size_t)(n > 0 ? n : 0), '\0');
  if (n > 0) {
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n,
                        nullptr, nullptr);
  }
  return s;
}

std::string path_to_utf8(const fs::path& p) { return wide_to_utf8(p.wstring()); }

std::string path_to_ansi(const fs::path& p) {
  const std::wstring w = p.wstring();
  if (w.empty()) return std::string();
  const int n = WideCharToMultiByte(CP_ACP, 0, w.data(), (int)w.size(), nullptr,
                                    0, nullptr, nullptr);
  std::string s((size_t)(n > 0 ? n : 0), '\0');
  if (n > 0) {
    WideCharToMultiByte(CP_ACP, 0, w.data(), (int)w.size(), s.data(), n,
                        nullptr, nullptr);
  }
  return s;
}

fs::path u8path(const std::string& u8) { return fs::path(utf8_to_wide(u8)); }

} // namespace gpt
