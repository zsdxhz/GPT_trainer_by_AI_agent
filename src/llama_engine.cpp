#include "llama_engine.h"

#include <winsock2.h>
#include <windows.h>
#include <winhttp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

#include "path_util.h"

#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "ws2_32.lib")

namespace fs = std::filesystem;

namespace gpt {

namespace {

// ---------------------------------------------------------------- 句柄封装
struct WinHttpHandle {
  HINTERNET h = nullptr;
  WinHttpHandle() = default;
  explicit WinHttpHandle(HINTERNET handle) : h(handle) {}
  ~WinHttpHandle() {
    if (h != nullptr) WinHttpCloseHandle(h);
  }
  WinHttpHandle(const WinHttpHandle&) = delete;
  WinHttpHandle& operator=(const WinHttpHandle&) = delete;
  HINTERNET get() const { return h; }
  explicit operator bool() const { return h != nullptr; }
};

std::string last_win_error(const char* what) {
  return std::string(what) + " 失败 (错误码 " + std::to_string(GetLastError()) + ")";
}

std::wstring widen(const std::string& s) { return utf8_to_wide(s); }

// ---------------------------------------------------------------- JSON 工具
std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += (char)c;
        }
    }
  }
  return out;
}

void append_utf8(std::string* out, unsigned int cp) {
  if (cp < 0x80) {
    out->push_back((char)cp);
  } else if (cp < 0x800) {
    out->push_back((char)(0xC0 | (cp >> 6)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back((char)(0xE0 | (cp >> 12)));
    out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out->push_back((char)(0xF0 | (cp >> 18)));
    out->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// 从 json 文本里取 "key":"..." 的字符串值（最小实现，够用于 llama-server 的 SSE）
bool json_extract_string(const std::string& json, const std::string& key,
                         std::string* out) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return false;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size() || json[pos] != '"') return false;
  ++pos;
  out->clear();
  while (pos < json.size()) {
    const char c = json[pos++];
    if (c == '"') return true;
    if (c != '\\') {
      out->push_back(c);
      continue;
    }
    if (pos >= json.size()) break;
    const char esc = json[pos++];
    switch (esc) {
      case '"': out->push_back('"'); break;
      case '\\': out->push_back('\\'); break;
      case '/': out->push_back('/'); break;
      case 'b': out->push_back('\b'); break;
      case 'f': out->push_back('\f'); break;
      case 'n': out->push_back('\n'); break;
      case 'r': out->push_back('\r'); break;
      case 't': out->push_back('\t'); break;
      case 'u': {
        unsigned int cp = 0;
        if (pos + 4 <= json.size()) {
          cp = (unsigned int)std::strtoul(json.substr(pos, 4).c_str(), nullptr, 16);
          pos += 4;
        }
        append_utf8(out, cp);
        break;
      }
      default: out->push_back(esc); break;
    }
  }
  return false;
}

bool json_extract_bool(const std::string& json, const std::string& key,
                       bool* out) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return false;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (json.compare(pos, 4, "true") == 0) {
    *out = true;
    return true;
  }
  if (json.compare(pos, 5, "false") == 0) {
    *out = false;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------- 端口
int pick_free_port() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(38000, 52000);
  for (int attempt = 0; attempt < 32; ++attempt) {
    const int port = dist(gen);
    SOCKET s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return port;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((u_short)port);
    const bool busy = (::bind(s, reinterpret_cast<sockaddr*>(&addr),
                              sizeof(addr)) != 0);
    ::closesocket(s);
    if (!busy) return port;
  }
  return 8080;
}

// 探测 llama-server 同目录下的 GPU 后端 DLL（Ollama 布局：cuda_v13/、cuda_v12/、vulkan/…）
void prepare_gpu_backends(const fs::path& server_exe) {
  std::error_code ec;
  wchar_t existing[1024] = {0};
  if (GetEnvironmentVariableW(L"GGML_BACKEND_PATH", existing, 1024) > 0) {
    return; // 用户已显式指定
  }
  const fs::path dir = server_exe.parent_path();
  struct Cand {
    const wchar_t* sub;
    const wchar_t* dll;
  };
  const Cand cands[] = {{L"cuda_v13", L"ggml-cuda.dll"},
                        {L"cuda_v12", L"ggml-cuda.dll"},
                        {L"vulkan", L"ggml-vulkan.dll"},
                        {L"rocm_v7_1", L"ggml-hip.dll"}};
  for (const auto& c : cands) {
    const fs::path dll = dir / c.sub / c.dll;
    if (fs::is_regular_file(dll, ec)) {
      SetEnvironmentVariableW(L"GGML_BACKEND_PATH", dll.c_str());
      return;
    }
  }
  // 后端 DLL 直接放在 exe 同目录
  for (const wchar_t* name : {L"ggml-cuda.dll", L"ggml-vulkan.dll"}) {
    const fs::path dll = dir / name;
    if (fs::is_regular_file(dll, ec)) {
      SetEnvironmentVariableW(L"GGML_BACKEND_PATH", dll.c_str());
      return;
    }
  }
}

std::string read_file_head(const fs::path& path, size_t max_bytes = 8192) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::string data(max_bytes, '\0');
  f.read(data.data(), (std::streamsize)data.size());
  data.resize((size_t)f.gcount());
  return data;
}

} // namespace

// ===========================================================================
// 构造 / 析构
// ===========================================================================
LlamaEngine::LlamaEngine() {
  WSADATA wsa{};
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
}

LlamaEngine::~LlamaEngine() { stop(); }

// ===========================================================================
// 探测 llama-server.exe
// ===========================================================================
fs::path LlamaEngine::detect_ollama_server() {
  std::vector<fs::path> candidates;
  wchar_t buf[MAX_PATH * 2] = {0};
  if (GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH * 2) > 0) {
    candidates.push_back(fs::path(buf) / "Programs" / "Ollama" / "lib" /
                         "ollama" / "llama-server.exe");
  }
  if (GetEnvironmentVariableW(L"ProgramFiles", buf, MAX_PATH * 2) > 0) {
    candidates.push_back(fs::path(buf) / "Ollama" / "lib" / "ollama" /
                         "llama-server.exe");
  }
  for (const auto& c : candidates) {
    std::error_code ec;
    if (fs::is_regular_file(c, ec)) return c;
  }
  return {};
}

fs::path LlamaEngine::detect_server_exe(const fs::path& hint) {
  std::error_code ec;
  if (!hint.empty() && fs::is_regular_file(hint, ec)) return hint;

  // 1) 项目内 third_party/llama.cpp（用户可自行放置 llama.cpp 发布包）
  std::vector<fs::path> candidates;
  const fs::path root = u8path("."); // 由调用方优先传 hint，这里只做兜底
  (void)root;
  const wchar_t* env_root = _wgetenv(L"GPT_ROOT");
  fs::path base = env_root != nullptr ? fs::path(env_root) : fs::path();
  if (!base.empty()) {
    candidates.push_back(base / "third_party" / "llama.cpp" / "llama-server.exe");
    candidates.push_back(base / "third_party" / "llama.cpp" / "bin" /
                         "llama-server.exe");
  }
  // 2) 可执行文件同目录 / 上层
  wchar_t exe_path[MAX_PATH * 2] = {0};
  if (GetModuleFileNameW(nullptr, exe_path, MAX_PATH * 2) > 0) {
    const fs::path exe_dir = fs::path(exe_path).parent_path();
    candidates.push_back(exe_dir / "llama-server.exe");
    candidates.push_back(exe_dir / "llama.cpp" / "llama-server.exe");
    candidates.push_back(exe_dir.parent_path() / "third_party" / "llama.cpp" /
                         "llama-server.exe");
  }
  for (const auto& c : candidates) {
    std::error_code ec2;
    if (fs::is_regular_file(c, ec2)) return c;
  }
  // 3) Ollama 自带
  if (const fs::path ollama = detect_ollama_server(); !ollama.empty()) {
    return ollama;
  }
  // 4) PATH
  wchar_t found[MAX_PATH * 2] = {0};
  if (SearchPathW(nullptr, L"llama-server.exe", nullptr, MAX_PATH * 2, found,
                  nullptr) > 0) {
    return fs::path(found);
  }
  return {};
}

// ===========================================================================
// HTTP 会话
// ===========================================================================
bool LlamaEngine::ensure_session(const std::string& host, int port) {
  close_session();
  WinHttpHandle session(WinHttpOpen(L"gpt-trainer/1.0",
                                    WINHTTP_ACCESS_TYPE_NO_PROXY,
                                    WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0));
  if (!session) {
    last_error_ = last_win_error("WinHttpOpen");
    return false;
  }
  WinHttpSetTimeouts(session.get(), 10000, 10000, 120000, 120000);
  WinHttpHandle connect(WinHttpConnect(session.get(), widen(host).c_str(),
                                       (INTERNET_PORT)port, 0));
  if (!connect) {
    last_error_ = last_win_error("WinHttpConnect");
    return false;
  }
  session_ = session.h;
  connect_ = connect.h;
  session.h = nullptr; // 移交所有权
  connect.h = nullptr;
  port_ = port;
  return true;
}

void LlamaEngine::close_session() {
  if (connect_ != nullptr) {
    WinHttpCloseHandle(static_cast<HINTERNET>(connect_));
    connect_ = nullptr;
  }
  if (session_ != nullptr) {
    WinHttpCloseHandle(static_cast<HINTERNET>(session_));
    session_ = nullptr;
  }
}

// 简易 GET（用于 /health）
static bool http_get(void* connect, const wchar_t* path, int* status,
                     std::string* body, std::string* err) {
  WinHttpHandle req(WinHttpOpenRequest(static_cast<HINTERNET>(connect),
                                       L"GET", path, nullptr,
                                       WINHTTP_NO_REFERER,
                                       WINHTTP_DEFAULT_ACCEPT_TYPES, 0));
  if (!req) {
    if (err) *err = "WinHttpOpenRequest 失败";
    return false;
  }
  if (!WinHttpSendRequest(req.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                          WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
    if (err) *err = "WinHttpSendRequest 失败";
    return false;
  }
  if (!WinHttpReceiveResponse(req.get(), nullptr)) {
    if (err) *err = "WinHttpReceiveResponse 失败";
    return false;
  }
  DWORD code = 0;
  DWORD len = sizeof(code);
  WinHttpQueryHeaders(req.get(),
                      WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                      WINHTTP_HEADER_NAME_BY_INDEX, &code, &len,
                      WINHTTP_NO_HEADER_INDEX);
  if (status) *status = (int)code;
  if (body != nullptr) {
    body->clear();
    for (;;) {
      DWORD avail = 0;
      if (!WinHttpQueryDataAvailable(req.get(), &avail) || avail == 0) break;
      std::string chunk(avail, '\0');
      DWORD read = 0;
      if (!WinHttpReadData(req.get(), chunk.data(), avail, &read)) break;
      chunk.resize(read);
      *body += chunk;
      if (body->size() > (1u << 20)) break;
    }
  }
  return true;
}

bool LlamaEngine::healthy() {
  if (connect_ == nullptr) return false;
  int status = 0;
  std::string body;
  std::string err;
  if (!http_get(connect_, L"/health", &status, &body, &err)) return false;
  return status == 200;
}

bool LlamaEngine::wait_until_ready(int timeout_ms, std::string* err) {
  const DWORD start = GetTickCount();
  while (GetTickCount() - start < (DWORD)timeout_ms) {
    if (process_ != nullptr) {
      const DWORD wait = WaitForSingleObject(static_cast<HANDLE>(process_), 0);
      if (wait == WAIT_OBJECT_0) {
        const std::string log = read_file_head(log_path_, 16384);
        if (err) {
          *err = "llama-server 启动后立即退出。日志:\n" + log;
        }
        return false;
      }
    }
    if (healthy()) return true;
    if (cancel_.load()) {
      if (err) *err = "启动已取消";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  const std::string log = read_file_head(log_path_, 16384);
  if (err) *err = "等待 llama-server 就绪超时。日志:\n" + log;
  return false;
}

// ===========================================================================
// 启动 / 停止
// ===========================================================================
bool LlamaEngine::start(const LlamaEngineOptions& opts, std::string* err) {
  auto fail = [&](const std::string& msg) {
    last_error_ = msg;
    if (err) *err = msg;
    return false;
  };

  // 统一转成绝对路径：子进程工作目录是 llama-server 所在目录，相对路径会解析错
  LlamaEngineOptions resolved = opts;
  {
    std::error_code ec;
    if (!resolved.server_exe.empty()) {
      resolved.server_exe = fs::absolute(resolved.server_exe, ec);
      ec.clear();
      if (!fs::is_regular_file(resolved.server_exe, ec)) {
        return fail("找不到 llama-server.exe，请在「高级」页设置 llama.cpp 路径");
      }
    } else {
      return fail("找不到 llama-server.exe，请在「高级」页设置 llama.cpp 路径");
    }
    if (resolved.model_path.empty()) {
      return fail("请选择 GGUF 模型文件");
    }
    resolved.model_path = fs::absolute(resolved.model_path, ec);
    ec.clear();
    if (!fs::is_regular_file(resolved.model_path, ec)) {
      return fail("找不到 GGUF 模型文件: " + path_to_utf8(resolved.model_path));
    }
  }

  // 已用同一模型运行则直接复用
  if (process_ != nullptr && resolved.model_path == opts_.model_path &&
      resolved.n_ctx == opts_.n_ctx &&
      resolved.n_gpu_layers == opts_.n_gpu_layers && healthy()) {
    return true;
  }
  stop();

  opts_ = resolved;
  cancel_.store(false);
  version_.clear();

  for (int attempt = 0; attempt < 3; ++attempt) {
    const int port = opts_.port > 0 ? opts_.port : pick_free_port();
    if (!ensure_session("127.0.0.1", port)) {
      close_session();
      continue;
    }

    // ---- 日志文件 ----
    wchar_t temp[MAX_PATH * 2] = {0};
    GetTempPathW(MAX_PATH * 2, temp);
    log_path_ = fs::path(temp) /
                ("gpt_trainer_llama_" + std::to_string(port) + ".log");

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE log_file = CreateFileW(log_path_.c_str(), GENERIC_WRITE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log_file == INVALID_HANDLE_VALUE) {
      last_error_ = "无法创建 llama-server 日志文件";
      close_session();
      continue;
    }

    // GPU 后端：Ollama 等发布包把 ggml-cuda.dll / ggml-vulkan.dll 放在子目录，
    // 需要通过 GGML_BACKEND_PATH 告诉 llama.cpp 去哪里加载
    if (opts_.n_gpu_layers > 0) prepare_gpu_backends(opts_.server_exe);

    // ---- 命令行 ----
    std::wostringstream args;
    args << L"\"" << opts_.server_exe.wstring() << L"\""
         << L" -m \"" << opts_.model_path.wstring() << L"\""
         << L" --host 127.0.0.1 --port " << port << L" -c " << opts_.n_ctx;
    if (opts_.n_threads > 0) args << L" -t " << opts_.n_threads;
    args << L" -ngl " << opts_.n_gpu_layers;
    if (opts_.batch > 0) args << L" -b " << opts_.batch;
    if (opts_.ubatch > 0) args << L" -ub " << opts_.ubatch;

    std::wstring cmd = args.str();
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = log_file;
    si.hStdError = log_file;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};

    const std::wstring cwd = opts_.server_exe.parent_path().wstring();
    const BOOL ok = CreateProcessW(opts_.server_exe.c_str(), cmd.data(), nullptr,
                                   nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                   cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    CloseHandle(log_file);
    if (!ok) {
      last_error_ = last_win_error("启动 llama-server");
      close_session();
      continue;
    }
    process_ = pi.hProcess;
    pid_ = pi.dwProcessId;
    CloseHandle(pi.hThread);

    // 让子进程随主程序退出（避免残留 llama-server）
    if (job_ == nullptr) {
      HANDLE job = CreateJobObjectW(nullptr, nullptr);
      if (job != nullptr) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        info.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info,
                                sizeof(info));
        AssignProcessToJobObject(job, pi.hProcess);
        job_ = job;
      }
    }

    std::string ready_err;
    if (wait_until_ready(180000, &ready_err)) {
      const std::string head = read_file_head(log_path_, 16384);
      std::istringstream iss(head);
      std::string line;
      while (std::getline(iss, line)) {
        if (line.find("build:") != std::string::npos ||
            line.find("version:") != std::string::npos) {
          version_ = line;
          break;
        }
      }
      last_error_.clear();
      refresh_model_props(); // 探测是否自带对话模板
      return true;
    }

    // 启动失败：清理后换端口重试
    stop();
    if (attempt == 2) return fail(ready_err);
  }
  return fail(last_error_.empty() ? "llama-server 启动失败" : last_error_);
}

void LlamaEngine::stop() {
  close_session();
  if (process_ != nullptr) {
    HANDLE h = static_cast<HANDLE>(process_);
    if (WaitForSingleObject(h, 0) != WAIT_OBJECT_0) {
      TerminateProcess(h, 0);
      WaitForSingleObject(h, 3000);
    }
    CloseHandle(h);
    process_ = nullptr;
    pid_ = 0;
  }
  has_chat_template_ = false;
  if (job_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(job_));
    job_ = nullptr;
  }
  port_ = 0;
}

// ===========================================================================
// 流式补全
// ===========================================================================
bool LlamaEngine::post_stream(const std::string& path, const std::string& body,
                              const DeltaFn& on_delta, std::string* out_text,
                              std::string* err, bool parse_sse) {
  (void)parse_sse;
  if (connect_ == nullptr) {
    if (err) *err = "llama-server 未启动";
    return false;
  }
  WinHttpHandle req(WinHttpOpenRequest(
      static_cast<HINTERNET>(connect_), L"POST", widen(path).c_str(), nullptr,
      WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0));
  if (!req) {
    if (err) *err = last_win_error("WinHttpOpenRequest");
    return false;
  }
  const wchar_t* headers = L"Content-Type: application/json\r\n";
  if (!WinHttpSendRequest(req.get(), headers, (DWORD)-1L, (LPVOID)body.data(),
                          (DWORD)body.size(), (DWORD)body.size(), 0)) {
    if (err) *err = last_win_error("WinHttpSendRequest");
    return false;
  }
  if (!WinHttpReceiveResponse(req.get(), nullptr)) {
    if (err) *err = last_win_error("WinHttpReceiveResponse");
    return false;
  }
  DWORD status = 0;
  DWORD len = sizeof(status);
  WinHttpQueryHeaders(req.get(),
                      WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                      WINHTTP_HEADER_NAME_BY_INDEX, &status, &len,
                      WINHTTP_NO_HEADER_INDEX);
  if (status != 200) {
    std::string error_body;
    for (;;) {
      DWORD avail = 0;
      if (!WinHttpQueryDataAvailable(req.get(), &avail) || avail == 0) break;
      std::string chunk(avail, '\0');
      DWORD read = 0;
      if (!WinHttpReadData(req.get(), chunk.data(), avail, &read)) break;
      chunk.resize(read);
      error_body += chunk;
      if (error_body.size() > 4096) break;
    }
    if (err) {
      *err = "llama-server 返回 HTTP " + std::to_string(status) + ": " +
             error_body;
    }
    return false;
  }

  std::string buffer;
  std::string line;
  bool finished = false;
  bool saw_data = false;
  while (!finished) {
    if (cancel_.load()) {
      if (err) *err = "已取消";
      break;
    }
    DWORD avail = 0;
    if (!WinHttpQueryDataAvailable(req.get(), &avail)) {
      if (err) *err = last_win_error("WinHttpQueryDataAvailable");
      return false;
    }
    if (avail == 0) break; // 流结束
    std::string chunk(avail, '\0');
    DWORD read = 0;
    if (!WinHttpReadData(req.get(), chunk.data(), avail, &read)) {
      if (err) *err = last_win_error("WinHttpReadData");
      return false;
    }
    buffer.append(chunk.data(), read);
    for (;;) {
      const size_t nl = buffer.find('\n');
      if (nl == std::string::npos) break;
      line = buffer.substr(0, nl);
      buffer.erase(0, nl + 1);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.rfind("data:", 0) != 0) continue;
      const std::string json = line.substr(5);
      if (json.find("[DONE]") != std::string::npos) {
        finished = true;
        break;
      }
      saw_data = true;
      std::string piece;
      if (json_extract_string(json, "content", &piece) && !piece.empty()) {
        if (out_text != nullptr) *out_text += piece;
        if (on_delta) on_delta(piece);
      }
      bool stop = false;
      if (json_extract_bool(json, "stop", &stop) && stop) {
        finished = true;
        break;
      }
      // OpenAI 风格（/v1/chat/completions）：finish_reason 非 null 即结束
      if (json.find("\"finish_reason\"") != std::string::npos &&
          json.find("\"finish_reason\":null") == std::string::npos) {
        finished = true;
        break;
      }
    }
  }
  if (!saw_data && out_text != nullptr && out_text->empty()) {
    if (err != nullptr && err->empty()) *err = "llama-server 没有返回任何内容";
    return false;
  }
  return true;
}

bool LlamaEngine::run_completion(const std::string& prompt_json,
                                  const LlamaSamplingParams& params,
                                  const DeltaFn& on_delta, std::string* out_text,
                                  std::string* err) {
  if (err) err->clear();
  if (out_text != nullptr) out_text->clear();
  if (connect_ == nullptr || !running()) {
    if (err) *err = "llama-server 未启动";
    return false;
  }
  if (!healthy()) {
    if (err) *err = "llama-server 无响应";
    return false;
  }
  cancel_.store(false);
  busy_.store(true);
  struct BusyGuard {
    std::atomic<bool>* flag;
    ~BusyGuard() { flag->store(false); }
  } guard{&busy_};

  std::string stop_json = "[";
  for (size_t i = 0; i < params.stop.size(); ++i) {
    if (i > 0) stop_json += ",";
    stop_json += "\"" + json_escape(params.stop[i]) + "\"";
  }
  stop_json += "]";

  std::ostringstream body;
  body << "{"
       << "\"prompt\":" << prompt_json << ","
       << "\"stream\":true,"
       << "\"temperature\":" << params.temperature << ","
       << "\"top_k\":" << params.top_k << ","
       << "\"top_p\":" << params.top_p << ","
       << "\"repeat_penalty\":" << params.repeat_penalty << ","
       << "\"n_predict\":" << params.max_tokens << ","
       << "\"seed\":" << (params.seed < 0 ? -1 : params.seed) << ","
       << "\"cache_prompt\":false,"
       << "\"n_keep\":0,"
       << "\"stop\":" << stop_json << "}";

  return post_stream("/completion", body.str(), on_delta, out_text, err, true);
}

bool LlamaEngine::complete(const std::string& prompt,
                           const LlamaSamplingParams& params,
                           const DeltaFn& on_delta, std::string* out_text,
                           std::string* err) {
  return run_completion("\"" + json_escape(prompt) + "\"", params, on_delta,
                        out_text, err);
}

bool LlamaEngine::complete_tokens(const std::vector<int32_t>& token_ids,
                                  const LlamaSamplingParams& params,
                                  const DeltaFn& on_delta, std::string* out_text,
                                  std::string* err) {
  std::string prompt_json = "[";
  for (size_t i = 0; i < token_ids.size(); ++i) {
    if (i > 0) prompt_json += ",";
    prompt_json += std::to_string(token_ids[i]);
  }
  prompt_json += "]";
  return run_completion(prompt_json, params, on_delta, out_text, err);
}

bool LlamaEngine::refresh_model_props() {
  has_chat_template_ = false;
  if (connect_ == nullptr) return false;
  int status = 0;
  std::string body;
  std::string err;
  if (!http_get(connect_, L"/props", &status, &body, &err)) return false;
  if (status != 200) return false;
  std::string tpl;
  if (json_extract_string(body, "chat_template", &tpl)) {
    // llama-server 用空串或 null 表示模型没有自带模板
    has_chat_template_ = !tpl.empty();
  }
  return true;
}

// 对话通道：/v1/chat/completions（由 llama.cpp 按模型 chat_template 套用格式）
bool LlamaEngine::complete_chat(const std::string& system_prompt,
                                const std::string& user_text,
                                const LlamaSamplingParams& params,
                                const DeltaFn& on_delta, std::string* out_text,
                                std::string* err) {
  if (err) err->clear();
  if (out_text != nullptr) out_text->clear();
  if (connect_ == nullptr || !running()) {
    if (err) *err = "llama-server 未启动";
    return false;
  }
  if (!healthy()) {
    if (err) *err = "llama-server 无响应";
    return false;
  }
  cancel_.store(false);
  busy_.store(true);
  struct BusyGuard {
    std::atomic<bool>* flag;
    ~BusyGuard() { flag->store(false); }
  } guard{&busy_};

  std::string stop_json = "[";
  for (size_t i = 0; i < params.stop.size(); ++i) {
    if (i > 0) stop_json += ",";
    stop_json += "\"" + json_escape(params.stop[i]) + "\"";
  }
  stop_json += "]";

  std::ostringstream body;
  body << "{\"messages\":[";
  if (!system_prompt.empty()) {
    body << "{\"role\":\"system\",\"content\":\""
         << json_escape(system_prompt) << "\"},";
  }
  body << "{\"role\":\"user\",\"content\":\"" << json_escape(user_text)
       << "\"}],"
       << "\"stream\":true,"
       << "\"temperature\":" << params.temperature << ","
       << "\"top_k\":" << params.top_k << ","
       << "\"top_p\":" << params.top_p << ","
       << "\"repeat_penalty\":" << params.repeat_penalty << ","
       << "\"max_tokens\":" << params.max_tokens << ","
       << "\"seed\":" << (params.seed < 0 ? -1 : params.seed) << ","
       << "\"cache_prompt\":false,"
       << "\"stop\":" << stop_json << "}";

  return post_stream("/v1/chat/completions", body.str(), on_delta, out_text, err,
                     true);
}

std::string LlamaEngine::model_info() {
  std::ostringstream oss;
  if (!running()) return "未启动";
  oss << path_to_utf8(opts_.model_path) << " | ctx=" << opts_.n_ctx
      << " | gpu_layers=" << opts_.n_gpu_layers
      << " | threads=" << (opts_.n_threads > 0 ? opts_.n_threads : 0)
      << " | port=" << port_;
  if (!version_.empty()) oss << "\n" << version_;
  return oss.str();
}

} // namespace gpt
