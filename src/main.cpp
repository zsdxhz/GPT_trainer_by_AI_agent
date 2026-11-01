// main.cpp - 启动入口
//   无参数 / gui         : 图形界面
//   <命令> [选项]        : 命令行模式（train / chat / tokenize / export-gguf /
//                          list / info / serve / selftest / loadtest / help）
//   --cli 等旧参数       : 仍然兼容（等价于对应子命令）
#include <torch/torch.h>

#include <QApplication>
#include <QStyleFactory>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <windows.h>
#include <shellapi.h>

#include "cli.h"
#include "config.h"
#include "gui.h"
#include "path_util.h"
#include "ui_theme.h"

namespace fs = std::filesystem;

// WIN32 子系统默认无控制台：附加到父进程控制台以便输出日志。
// 仅在标准输出句柄无效时重定向到控制台（不破坏管道/文件重定向）。
static void attach_console() {
  if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
  HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
  if (h == nullptr || h == INVALID_HANDLE_VALUE) {
    FILE* fp = nullptr;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    freopen_s(&fp, "CONOUT$", "w", stderr);
  }
  // 中文命令行/日志统一用 UTF-8
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
}

// 以宽字符重新解析命令行（中文/空格路径不会因 ANSI 编码而失效）
static std::vector<std::wstring> wide_args() {
  std::vector<std::wstring> out;
  int argc = 0;
  LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
  if (argv == nullptr) return out;
  for (int i = 0; i < argc; ++i) out.emplace_back(argv[i]);
  LocalFree(argv);
  return out;
}

int main(int argc, char** argv) {
  attach_console();

  const std::vector<std::wstring> wargs = wide_args();
  std::wstring first;
  if (wargs.size() > 1) {
    first = wargs[1];
    for (auto& c : first) c = (wchar_t)towlower(c);
  }
  const bool has_arg = !first.empty();
  const bool wants_gui =
      !has_arg || first == L"gui" || first == L"--gui" || first == L"-gui";

  if (has_arg && !wants_gui) {
    if (gpt::is_cli_command(first)) {
      try {
        return gpt::run_cli(wargs);
      } catch (const std::exception& e) {
        std::cerr << "命令执行失败: " << e.what() << std::endl;
        return 1;
      }
    }
    if (first[0] != L'-') {
      std::cerr << "未知命令: " << gpt::wide_to_utf8(wargs[1]) << "\n"
                << "运行 gpt_trainer.exe help 查看可用命令。" << std::endl;
      return 2;
    }
  }

  // ---- GUI 模式 ----
  QApplication app(argc, argv);

  // 兜底：任何未捕获异常/terminate 都写日志并弹窗，而不是静默闪退
  std::set_terminate([] {
    std::string msg = "未知错误";
    if (auto eptr = std::current_exception()) {
      try {
        std::rethrow_exception(eptr);
      } catch (const std::exception& e) {
        msg = e.what();
      } catch (...) {
        msg = "非 std::exception 异常";
      }
    }
    std::wstring tmp(MAX_PATH * 2, L'\0');
    const DWORD n = GetTempPathW((DWORD)tmp.size(), tmp.data());
    if (n > 0) {
      tmp.resize(n);
      std::ofstream f(std::filesystem::path(tmp) / L"gpt_trainer_terminate.log",
                      std::ios::app);
      f << "[terminate] " << msg << std::endl;
    }
    MessageBoxW(nullptr,
                (L"程序遇到未处理的错误：\n" + std::wstring(msg.begin(), msg.end()) +
                 L"\n\n详细信息已写入 %TEMP%\\gpt_trainer_terminate.log")
                    .c_str(),
                L"GPT 训练与对话系统", MB_ICONERROR | MB_OK);
    std::abort();
  });

  // 原生 Windows 11 (Fluent) 控件风格：由 Qt 的 qmodernwindowsstyle 提供。
  if (QStyle* style = QStyleFactory::create(QStringLiteral("windows11"))) {
    app.setStyle(style);
  } else if (QStyle* fusion = QStyleFactory::create(QStringLiteral("Fusion"))) {
    app.setStyle(fusion);
  }

  // 安装 Fluent 主题：字体、跟随系统浅色/深色 + 强调色、全局样式、DWM 装饰
  fl::Theme::install();

  try {
    gpt::TrainingGUI gui;
    gui.show();
    return app.exec();
  } catch (const std::exception& e) {
    std::cerr << "GUI 初始化失败: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "GUI 初始化失败: 未知异常" << std::endl;
    return 1;
  }
}
