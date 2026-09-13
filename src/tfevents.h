#pragma once
// tfevents.h - 手写 TensorBoard events 文件写出器（对应 python 的 SummaryWriter）
// 输出格式与新版 tensorboard (record_writer.py) 完全一致：
//   [8 字节 LE 长度][4 字节 masked CRC32C(长度)][protobuf 载荷][4 字节 masked CRC32C(载荷)]
// 载荷本身不做掩码。
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gpt {

class SummaryWriter {
 public:
  // 在 log_dir 下创建 events.out.tfevents.<时间>.<主机名>.<pid>.0
  explicit SummaryWriter(const std::filesystem::path& log_dir);
  ~SummaryWriter();

  SummaryWriter(const SummaryWriter&) = delete;
  SummaryWriter& operator=(const SummaryWriter&) = delete;

  void add_scalar(const std::string& tag, float value, int64_t step);
  void close();

 private:
  void write_event(const std::string& payload);

  std::string filename_;
  std::ofstream out_;
  bool closed_ = false;
};

} // namespace gpt
