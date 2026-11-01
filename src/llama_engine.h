#pragma once
// llama_engine.h - llama.cpp 推理引擎封装
//
// 实现方式：以 llama.cpp 官方 llama-server（llama-server.exe）作为推理后端，
//   进程间用 HTTP + SSE 流式传输。这样做的取舍：
//   * 直接复用 llama.cpp 成熟的内核与多后端（CPU / CUDA / Vulkan / Metal…），
//     不需要在本项目里编译 ggml，也不绑定 llama.cpp 的 C ABI（跨版本更稳）
//   * 崩溃隔离：推理进程异常不会带走训练主程序
//   * 模型支持：任意 llama.cpp 支持的 GGUF（本项目的模型可导出为 gpt2 架构 GGUF）
//
// llama-server.exe 探测顺序：显式指定 → <项目根>/third_party/llama.cpp/llama-server.exe
//   → Ollama 安装目录 → PATH
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gpt {

struct LlamaEngineOptions {
  std::filesystem::path server_exe; // llama-server.exe
  std::filesystem::path model_path; // .gguf
  int n_ctx = 2048;                 // 上下文长度
  int n_threads = 0;                // 0 = 自动（物理核数）
  int n_gpu_layers = 99;            // 99 = 尽量全部放到 GPU；0 = 纯 CPU
  int port = 0;                     // 0 = 自动挑一个空闲端口
  int batch = 512;
  int ubatch = 256;
  bool verbose = false;             // 保留 llama-server 详细日志
};

struct LlamaSamplingParams {
  double temperature = 0.8;
  int top_k = 40;
  double top_p = 0.95;
  double repeat_penalty = 1.1;
  int max_tokens = 128;
  int seed = -1; // -1 = 随机
  std::vector<std::string> stop;
};

class LlamaEngine {
 public:
  using DeltaFn = std::function<void(const std::string&)>;

  LlamaEngine();
  ~LlamaEngine();
  LlamaEngine(const LlamaEngine&) = delete;
  LlamaEngine& operator=(const LlamaEngine&) = delete;

  // 启动（或复用已启动的）llama-server，阻塞直到 /health 就绪或超时
  bool start(const LlamaEngineOptions& opts, std::string* err);
  void stop();
  bool running() const { return process_ != nullptr; }
  bool healthy();

  // 当前加载的模型 / 上下文信息（失败返回空）
  std::string model_info();

  // 流式补全：prompt 为原始文本；on_delta 可能在线程外被调用
  bool complete(const std::string& prompt, const LlamaSamplingParams& params,
                const DeltaFn& on_delta, std::string* out_text, std::string* err);
  // 流式补全（token 通道）：直接用 token id 数组作为 prompt。
  // 本项目导出的 GGUF 词表是中文 SentencePiece，llama.cpp 自带分词器可能
  // 无法处理拉丁字符（会返回 400），因此推荐先用 sentencepiece 分词再走这里。
  bool complete_tokens(const std::vector<int32_t>& token_ids,
                       const LlamaSamplingParams& params, const DeltaFn& on_delta,
                       std::string* out_text, std::string* err);
  // 流式补全（对话通道）：走 /v1/chat/completions，由 llama.cpp 按模型自带
  // chat_template（ChatML 等）套用指令格式，Qwen/Llama-Instruct 等模型应走这里
  bool complete_chat(const std::string& system_prompt,
                     const std::string& user_text,
                     const LlamaSamplingParams& params, const DeltaFn& on_delta,
                     std::string* out_text, std::string* err);
  // 该模型是否自带对话模板（/props 的 chat_template 非空）
  bool has_chat_template() const { return has_chat_template_; }
  // 读取 /props 刷新 chat_template 状态（start() 成功后自动调用）
  bool refresh_model_props();
  void cancel() { cancel_.store(true); }

  bool busy() const { return busy_.load(); }
  const std::string& last_error() const { return last_error_; }
  const std::filesystem::path& log_path() const { return log_path_; }
  const LlamaEngineOptions& options() const { return opts_; }
  // llama-server --version 的第一行（启动后可用）
  std::string version_note() const { return version_; }

  // 探测 llama-server.exe 路径（找不到返回空）
  static std::filesystem::path detect_server_exe(
      const std::filesystem::path& hint = {});
  // 猜测 Ollama 自带的 llama.cpp 位置（找不到返回空）
  static std::filesystem::path detect_ollama_server();

 private:
  bool ensure_session(const std::string& host, int port);
  void close_session();
  bool wait_until_ready(int timeout_ms, std::string* err);
  bool post_stream(const std::string& path, const std::string& body,
                   const DeltaFn& on_delta, std::string* out_text,
                   std::string* err, bool parse_sse);
  // prompt_json 是 Json 片段：文本为 "xxx"，token 为 [1,2,3]
  bool run_completion(const std::string& prompt_json,
                      const LlamaSamplingParams& params, const DeltaFn& on_delta,
                      std::string* out_text, std::string* err);

  void* process_ = nullptr; // HANDLE
  unsigned long pid_ = 0;
  void* job_ = nullptr; // HANDLE（用于随主程序退出而结束子进程）
  void* session_ = nullptr;
  void* connect_ = nullptr;
  std::filesystem::path log_path_;
  LlamaEngineOptions opts_;
  std::string version_;
  std::string last_error_;
  bool has_chat_template_ = false;
  std::atomic<bool> cancel_{false};
  std::atomic<bool> busy_{false};
  int port_ = 0;
};

} // namespace gpt
