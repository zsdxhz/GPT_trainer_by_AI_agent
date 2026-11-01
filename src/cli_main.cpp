#include "cli.h"

#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace gpt {

namespace {

std::wstring strip_dashes(std::wstring s) {
  while (!s.empty() && s[0] == L'-') s.erase(s.begin());
  return s;
}

const wchar_t* kVersion = L"1.1.0";

void print_version() {
  std::cout << "gpt_trainer (C++ GPT 训练与对话系统) " << cli_utf8(kVersion)
            << "\n  - 训练/分词器: libtorch + SentencePiece（进程内）"
            << "\n  - 对话推理  : llama.cpp（llama-server 子进程，可回退内置引擎）"
            << "\n  - 界面      : Windows 11 Fluent（GUI 模式直接运行本程序）"
            << std::endl;
}

} // namespace

// ===========================================================================
// 命令注册
// ===========================================================================
bool is_cli_command(const std::wstring& name) {
  static const std::set<std::wstring> commands = {
      L"train",      L"chat",       L"tokenize", L"export-gguf",
      L"list",       L"info",       L"serve",    L"selftest",
      L"loadtest",   L"help",       L"version",  L"cli",
      L"llama-chat",
  };
  return commands.count(strip_dashes(name)) > 0;
}

// ===========================================================================
// 帮助
// ===========================================================================
void cli_print_help() {
  std::cout <<
      "gpt_trainer — C++ GPT 训练与对话系统（命令行）\n"
      "\n"
      "用法: gpt_trainer.exe <命令> [选项]\n"
      "      不带参数直接运行 = 打开图形界面（GUI）\n"
      "\n"
      "命令:\n"
      "  train         训练模型（数据/结构/优化/轮次/断点全部可调）\n"
      "  chat          对话或续写（llama.cpp / 内置引擎，支持交互式 REPL 与流式输出）\n"
      "  tokenize      训练 SentencePiece 分词器并打包 dataset.zip\n"
      "  export-gguf   把 saves 模型导出为 GGUF（llama.cpp 可直接加载）\n"
      "  list          列出训练模型与 GGUF（含体积）\n"
      "  info          设备 / 路径 / 引擎探测信息\n"
      "  serve         只启动 llama-server（调试/被其他程序调用）\n"
      "  selftest      互读自检（加载 Python 训练的模型 → 前向 → 生成 → 回环）\n"
      "  loadtest      加载任意 state_dict 并打印张量\n"
      "  help          显示本帮助（也可用 <命令> --help 看子命令选项）\n"
      "  version       版本信息\n"
      "\n"
      "全局选项:\n"
      "  -h, --help    帮助        -q, --quiet   安静模式（只输出结果）\n"
      "  --json        机器可读输出（适合脚本/Agent 调用）\n"
      "\n"
      "常用示例:\n"
      "  gpt_trainer.exe train --epochs 3 --batch 8\n"
      "  gpt_trainer.exe tokenize --out D:\\data corpus.txt\n"
      "  gpt_trainer.exe export-gguf --model saves/model1\n"
      "  gpt_trainer.exe chat --model saves/model1 --prompt \"今天天气\"\n"
      "  gpt_trainer.exe chat --model models/gguf/model1.gguf --interactive\n"
      "  gpt_trainer.exe list --json\n"
      "  gpt_trainer.exe info\n"
      "\n"
      "提示：所有命令都可以加 --json，方便被脚本或其它程序解析。\n";
}

// ===========================================================================
// 入口
// ===========================================================================
int run_cli(const std::vector<std::wstring>& argv) {
  if (argv.size() < 2) {
    cli_print_help();
    return 0;
  }
  const std::wstring raw_cmd = argv[1];
  const std::wstring cmd = strip_dashes(raw_cmd);
  const bool legacy = !raw_cmd.empty() && raw_cmd[0] == L'-';

  CliArgs a = parse_cli(argv, 2);
  cli_set_json(a.has(L"json"));
  cli_set_quiet(a.has(L"quiet"));

  if (cmd == L"version") {
    print_version();
    return 0;
  }
  if (cmd == L"help" || (a.has(L"help") && cmd.empty())) {
    cli_print_help();
    return 0;
  }
  if (cmd == L"train" || cmd == L"cli") return cli_cmd_train(a, legacy);
  if (cmd == L"chat" || cmd == L"llama-chat") {
    return cli_cmd_chat(a, legacy || cmd == L"llama-chat");
  }
  if (cmd == L"tokenize") return cli_cmd_tokenize(a, legacy);
  if (cmd == L"export-gguf") return cli_cmd_export_gguf(a, legacy);
  if (cmd == L"list") return cli_cmd_list(a);
  if (cmd == L"info") return cli_cmd_info(a);
  if (cmd == L"serve") return cli_cmd_serve(a);
  if (cmd == L"selftest") return cli_cmd_selftest(a);
  if (cmd == L"loadtest") return cli_cmd_loadtest(a);

  cli_error("未知命令: " + cli_utf8(raw_cmd));
  cli_print_help();
  return 2;
}

} // namespace gpt
