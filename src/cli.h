#pragma once
// cli.h - 命令行接口（与 GUI 功能对应的完整 CLI）
//
// 命令：
//   train        训练模型（全部超参可调）
//   chat         对话 / 续写（llama.cpp 或内置引擎，支持交互式 REPL 与流式输出）
//   tokenize     训练 SentencePiece 分词器并打包 dataset.zip
//   export-gguf  把 saves 模型导出为 GGUF
//   list         列出训练模型与 GGUF
//   info         设备 / 路径 / 引擎信息
//   serve        只启动 llama-server（调试用）
//   selftest     互读自检
//   loadtest     加载任意 state_dict 并打印张量
//   help         帮助
//
// 全局选项：-h/--help  --version  -q/--quiet  --json  -v/--verbose
// 兼容旧参数：--cli / --selftest / --loadtest / --tokenize / --export-gguf / --llama-chat
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gpt {

// ---------------------------------------------------------------------------
// 参数解析结果（支持 --key value / --key=value / -k value / --flag）
// ---------------------------------------------------------------------------
struct CliArgs {
  std::vector<std::wstring> positional;
  std::map<std::wstring, std::vector<std::wstring>> opts;

  bool has(const std::wstring& key) const;
  std::wstring get(const std::wstring& key, const std::wstring& def = L"") const;
  long long get_int(const std::wstring& key, long long def) const;
  double get_double(const std::wstring& key, double def) const;
  std::vector<std::wstring> get_list(const std::wstring& key) const;
  std::wstring pos(size_t index, const std::wstring& def = L"") const;
};

// 解析 argv（从 start 下标开始），短选项会展开为长选项
CliArgs parse_cli(const std::vector<std::wstring>& argv, size_t start);

// ---------------------------------------------------------------------------
// 输出辅助（统一处理 --json / --quiet / UTF-8 控制台）
// ---------------------------------------------------------------------------
void cli_set_quiet(bool quiet);
void cli_set_json(bool json);
bool cli_quiet();
bool cli_json();
// 普通信息行（--quiet 时丢弃）
void cli_say(const std::string& line);
// 错误行（写 stderr）
void cli_error(const std::string& line);
// JSON 字符串转义（含引号）
std::string cli_json_escape(const std::string& s);
// 宽字符 → UTF-8
std::string cli_utf8(const std::wstring& w);
// 当前时间 "HH:mm:ss"
std::string cli_time();

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
// argv[0]=程序名, argv[1]=命令；返回进程退出码（0 成功 / 1 运行失败 / 2 用法错误）
int run_cli(const std::vector<std::wstring>& argv);
// 是否是已知命令（含旧式 --xxx 形式）
bool is_cli_command(const std::wstring& name);
// 打印总帮助
void cli_print_help();
// 子命令帮助（cli.cpp / cli_chat.cpp 各自实现）
void cli_print_chat_help();
void cli_print_selftest_help();
void cli_print_loadtest_help();

// 子命令（cli.cpp / cli_chat.cpp 内实现）
int cli_cmd_train(const CliArgs& a, bool legacy);
int cli_cmd_chat(const CliArgs& a, bool legacy);
int cli_cmd_tokenize(const CliArgs& a, bool legacy);
int cli_cmd_export_gguf(const CliArgs& a, bool legacy);
int cli_cmd_list(const CliArgs& a);
int cli_cmd_info(const CliArgs& a);
int cli_cmd_serve(const CliArgs& a);
int cli_cmd_selftest(const CliArgs& a);
int cli_cmd_loadtest(const CliArgs& a);

// 训练/分词器共用的语料目录（<项目根>/data/data/raw）
std::wstring cli_default_raw_dir();

} // namespace gpt
