#include "tfevents.h"

#include <windows.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace gpt {

// ---------------------------------------------------------------------------
// CRC32C（Castagnoli，与 tensorflow crc32c.cc 一致）
// ---------------------------------------------------------------------------
namespace {

uint32_t g_crc_table[256];
bool g_crc_ready = []() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int j = 0; j < 8; ++j) {
      crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
    }
    g_crc_table[i] = crc;
  }
  return true;
}();

uint32_t crc32c(const void* data, size_t len) {
  const uint8_t* p = (const uint8_t*)data;
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc = g_crc_table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  }
  return ~crc;
}

// tensorflow crc32c::Mask
uint32_t mask_crc(uint32_t crc) {
  return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}

// protobuf 编码辅助
void append_varint(std::string& out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back((char)((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back((char)v);
}

void append_tag(std::string& out, int field, int wire) {
  append_varint(out, (uint64_t)((field << 3) | wire));
}

void append_bytes(std::string& out, int field, const std::string& data) {
  append_tag(out, field, 2);
  append_varint(out, data.size());
  out += data;
}

void append_fixed64(std::string& out, int field, double v) {
  append_tag(out, field, 1);
  uint64_t bits;
  std::memcpy(&bits, &v, 8);
  for (int i = 0; i < 8; ++i) out.push_back((char)((bits >> (8 * i)) & 0xFF));
}

void append_fixed32(std::string& out, int field, float v) {
  append_tag(out, field, 5);
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  for (int i = 0; i < 4; ++i) out.push_back((char)((bits >> (8 * i)) & 0xFF));
}

void append_int64(std::string& out, int field, int64_t v) {
  append_tag(out, field, 0);
  append_varint(out, (uint64_t)v);
}

} // namespace

// ---------------------------------------------------------------------------
// SummaryWriter
// ---------------------------------------------------------------------------
SummaryWriter::SummaryWriter(const std::filesystem::path& log_dir) {
  uint64_t now = (uint64_t)std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  char host[256] = {};
  DWORD host_len = sizeof(host);
  if (!GetComputerNameA(host, &host_len)) std::strcpy(host, "localhost");
  DWORD pid = GetCurrentProcessId();
  const std::string fname = "events.out.tfevents." + std::to_string(now) + "." +
                            host + "." + std::to_string(pid) + ".0";
  filename_ = (log_dir / fname).string();
  out_.open(filename_, std::ios::binary | std::ios::trunc);
  if (!out_) throw std::runtime_error("无法创建 TensorBoard 日志: " + filename_);

  // 首条记录：file_version
  std::string payload;
  append_bytes(payload, 3, "brain.Event:2");
  write_event(payload);
}

SummaryWriter::~SummaryWriter() { close(); }

void SummaryWriter::close() {
  if (!closed_) {
    out_.flush();
    out_.close();
    closed_ = true;
  }
}

void SummaryWriter::write_event(const std::string& payload) {
  if (closed_) return;
  const uint64_t len = payload.size();
  // 头部：8 字节 LE 长度 + masked CRC32C(长度字节)
  char hdr[12];
  std::memcpy(hdr, &len, 8);
  uint32_t hcrc = mask_crc(crc32c(&len, 8));
  std::memcpy(hdr + 8, &hcrc, 4);
  out_.write(hdr, 12);
  // 载荷：不掩码
  out_.write(payload.data(), (std::streamsize)payload.size());
  // 尾部：masked CRC32C(载荷)
  uint32_t dcrc = mask_crc(crc32c(payload.data(), payload.size()));
  out_.write((const char*)&dcrc, 4);
  out_.flush();
}

void SummaryWriter::add_scalar(const std::string& tag, float value, int64_t step) {
  if (closed_) return;
  double wall = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count() / 1e9;

  // Summary.Value { tag=1, simple_value=2 }
  std::string value_msg;
  append_bytes(value_msg, 1, tag);
  append_fixed32(value_msg, 2, value);
  // Summary { value=1 }
  std::string summary_msg;
  append_bytes(summary_msg, 1, value_msg);
  // Event { wall_time=1, step=2, summary=5 }
  std::string event_msg;
  append_fixed64(event_msg, 1, wall);
  append_int64(event_msg, 2, step);
  append_bytes(event_msg, 5, summary_msg);

  write_event(event_msg);
}

} // namespace gpt
