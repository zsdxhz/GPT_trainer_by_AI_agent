#include "zip_util.h"

#include <zlib.h>

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "path_util.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

// ---- ZIP 格式常量 ----
constexpr uint32_t kEocdSig = 0x06054b50; // End of central directory
constexpr uint32_t kCdSig = 0x02014b50;   // Central directory file header
constexpr uint32_t kLhSig = 0x04034b50;   // Local file header
constexpr uint16_t kFlagDataDescriptor = 0x0008;
constexpr uint16_t kFlagUtf8 = 0x0800;

uint16_t rd16(const std::vector<uint8_t>& b, size_t off) {
  return (uint16_t)(b[off] | (b[off + 1] << 8));
}
uint32_t rd32(const std::vector<uint8_t>& b, size_t off) {
  return (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8) |
         ((uint32_t)b[off + 2] << 16) | ((uint32_t)b[off + 3] << 24);
}

struct Entry {
  std::string name;
  uint16_t method = 0;
  uint16_t flags = 0;
  uint32_t comp_size = 0;
  uint32_t uncomp_size = 0;
  uint32_t local_offset = 0;
};

// 路径安全：拒绝绝对路径与 ..
bool safe_name(const std::string& name) {
  fs::path p(name);
  if (p.is_absolute()) return false;
  for (const auto& part : p) {
    if (part == "..") return false;
  }
  return true;
}

} // namespace

bool extract_zip(const fs::path& zip_path, const fs::path& out_dir,
                 std::string* err) {
  try {
    // 读入整个文件
    std::ifstream in(zip_path, std::ios::binary);
    if (!in) {
      if (err) *err = "无法打开 " + path_to_utf8(zip_path);
      return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    in.close();
    if (buf.size() < 22) {
      if (err) *err = "文件过小，不是合法 zip";
      return false;
    }

    // 定位 EOCD（从尾部向前扫描，最多 65557 字节）
    size_t eocd = std::string::npos;
    size_t scan_from = buf.size() > 65557 ? buf.size() - 65557 : 0;
    for (size_t i = buf.size() - 22 + 1; i-- > scan_from;) {
      if (rd32(buf, i) == kEocdSig) {
        // 注释长度必须精确匹配到文件末尾
        uint16_t comment_len = rd16(buf, i + 20);
        if (i + 22 + comment_len == buf.size()) {
          eocd = i;
          break;
        }
      }
      if (i == 0) break;
    }
    if (eocd == std::string::npos) {
      if (err) *err = "未找到 EOCD，不是合法 zip";
      return false;
    }

    uint32_t cd_offset = rd32(buf, eocd + 16);
    uint16_t entry_count = rd16(buf, eocd + 10);
    if (cd_offset + 46 > buf.size()) {
      if (err) *err = "中央目录偏移非法";
      return false;
    }

    // 解析中央目录
    std::vector<Entry> entries;
    size_t p = cd_offset;
    for (uint16_t i = 0; i < entry_count; ++i) {
      if (p + 46 > buf.size() || rd32(buf, p) != kCdSig) {
        if (err) *err = "中央目录项损坏";
        return false;
      }
      Entry e;
      e.method = rd16(buf, p + 10);
      e.flags = rd16(buf, p + 8);
      e.comp_size = rd32(buf, p + 20);
      e.uncomp_size = rd32(buf, p + 24);
      uint16_t name_len = rd16(buf, p + 28);
      uint16_t extra_len = rd16(buf, p + 30);
      uint16_t comment_len = rd16(buf, p + 32);
      e.local_offset = rd32(buf, p + 42);
      if (p + 46 + name_len > buf.size()) {
        if (err) *err = "文件名超出文件范围";
        return false;
      }
      std::string raw_name((const char*)&buf[p + 46], name_len);
      // UTF-8 标志位（python zipfile 对非 ASCII 文件名会设置）
      e.name = (e.flags & kFlagUtf8) ? path_to_utf8(u8path(raw_name))
                                     : raw_name;
      entries.push_back(std::move(e));
      p += 46 + name_len + extra_len + comment_len;
    }

    fs::create_directories(out_dir);

    // 逐个解压
    for (const auto& e : entries) {
      if (e.name.empty() || e.name.back() == '/') {
        if (!e.name.empty()) fs::create_directories(out_dir / u8path(e.name));
        continue;
      }
      if (!safe_name(e.name)) {
        if (err) *err = "跳过不安全的路径: " + e.name;
        return false;
      }
      if (e.local_offset + 30 > buf.size() || rd32(buf, e.local_offset) != kLhSig) {
        if (err) *err = "本地文件头损坏: " + e.name;
        return false;
      }
      uint16_t lname_len = rd16(buf, e.local_offset + 26);
      uint16_t lextra_len = rd16(buf, e.local_offset + 28);
      size_t data_off = e.local_offset + 30 + lname_len + lextra_len;
      uint32_t comp_size = e.comp_size;
      uint32_t uncomp_size = e.uncomp_size;
      // 若本地头带 data descriptor 标志，尺寸以中央目录为准（此处已按中央目录取）
      if (data_off + comp_size > buf.size()) {
        if (err) *err = "数据超出文件范围: " + e.name;
        return false;
      }

      fs::path out_path = out_dir / u8path(e.name);
      fs::create_directories(out_path.parent_path());
      std::ofstream os(out_path, std::ios::binary);
      if (!os) {
        if (err) *err = "无法写入 " + path_to_utf8(out_path);
        return false;
      }

      if (e.method == 0) { // store
        os.write((const char*)&buf[data_off], comp_size);
      } else if (e.method == 8) { // deflate
        std::vector<char> out(uncomp_size);
        z_stream zs{};
        if (inflateInit2_(&zs, -15, ZLIB_VERSION, (int)sizeof(z_stream)) != Z_OK) {
          if (err) *err = "inflateInit 失败: " + e.name;
          return false;
        }
        zs.next_in = &buf[data_off];
        zs.avail_in = (uInt)comp_size;
        zs.next_out = (Bytef*)out.data();
        zs.avail_out = (uInt)uncomp_size;
        int r = inflate(&zs, Z_FINISH);
        inflateEnd(&zs);
        if (r != Z_STREAM_END) {
          if (err) *err = "inflate 失败(" + std::to_string(r) + "): " + e.name;
          return false;
        }
        os.write(out.data(), (std::streamsize)out.size());
      } else {
        if (err) *err = "不支持的压缩方式 " + std::to_string(e.method) + ": " + e.name;
        return false;
      }
      os.close();
    }
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

// ---------------------------------------------------------------------------
// 打包（deflate 压缩）
// ---------------------------------------------------------------------------
namespace {

void wr16(std::string& out, uint16_t v) {
  out.push_back((char)(v & 0xFF));
  out.push_back((char)((v >> 8) & 0xFF));
}
void wr32(std::string& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back((char)((v >> (8 * i)) & 0xFF));
}

// DOS 时间戳（zip 本地/中央目录头用）
void dos_time(uint16_t& t, uint16_t& d) {
  SYSTEMTIME st;
  GetLocalTime(&st);
  t = (uint16_t)((st.wHour << 11) | (st.wMinute << 5) | (st.wSecond / 2));
  d = (uint16_t)(((st.wYear - 1980) << 9) | (st.wMonth << 5) | st.wDay);
}

struct ZipOutEntry {
  std::string name;
  uint32_t crc = 0;
  uint32_t comp_size = 0;
  uint32_t uncomp_size = 0;
  uint32_t offset = 0;
};

// raw deflate（zip 格式：无 zlib 头）
bool deflate_raw(const std::string& in, std::string* out, std::string* err) {
  z_stream zs{};
  if (deflateInit2_(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
                    Z_DEFAULT_STRATEGY, ZLIB_VERSION,
                    (int)sizeof(z_stream)) != Z_OK) {
    if (err) *err = "deflateInit 失败";
    return false;
  }
  out->resize(deflateBound(&zs, (uLong)in.size()));
  zs.next_in = (Bytef*)in.data();
  zs.avail_in = (uInt)in.size();
  zs.next_out = (Bytef*)out->data();
  zs.avail_out = (uInt)out->size();
  const int r = deflate(&zs, Z_FINISH);
  out->resize(zs.total_out);
  deflateEnd(&zs);
  if (r != Z_STREAM_END) {
    if (err) *err = "deflate 失败(" + std::to_string(r) + ")";
    return false;
  }
  return true;
}

} // namespace

bool create_zip(
    const fs::path& zip_path,
    const std::vector<std::pair<std::string, fs::path>>& entries,
    std::string* err) {
  try {
    std::ofstream os(zip_path, std::ios::binary | std::ios::trunc);
    if (!os) {
      if (err) *err = "无法创建 " + path_to_utf8(zip_path);
      return false;
    }

    std::vector<ZipOutEntry> outs;
    for (const auto& [name, file] : entries) {
      if (name.empty() || !safe_name(name)) {
        if (err) *err = "非法 zip 条目名: " + name;
        return false;
      }
      std::ifstream in(file, std::ios::binary);
      if (!in) {
        if (err) *err = "无法读取 " + path_to_utf8(file);
        return false;
      }
      std::string raw((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
      in.close();

      std::string comp;
      if (!deflate_raw(raw, &comp, err)) return false;

      ZipOutEntry e;
      e.name = name;
      e.crc = (uint32_t)crc32(0, (const Bytef*)raw.data(), (uInt)raw.size());
      e.comp_size = (uint32_t)comp.size();
      e.uncomp_size = (uint32_t)raw.size();
      e.offset = (uint32_t)os.tellp();

      uint16_t tm = 0, dt = 0;
      dos_time(tm, dt);

      // ---- 本地文件头 ----
      std::string lh;
      wr32(lh, kLhSig);
      wr16(lh, 20);                    // version needed
      wr16(lh, kFlagUtf8);             // UTF-8 文件名
      wr16(lh, 8);                     // deflate
      wr16(lh, tm);
      wr16(lh, dt);
      wr32(lh, e.crc);
      wr32(lh, e.comp_size);
      wr32(lh, e.uncomp_size);
      wr16(lh, (uint16_t)name.size());
      wr16(lh, 0);                     // extra len
      lh += name;
      os.write(lh.data(), (std::streamsize)lh.size());
      os.write(comp.data(), (std::streamsize)comp.size());
      outs.push_back(std::move(e));
    }

    const uint32_t cd_offset = (uint32_t)os.tellp();
    std::string cd;
    for (const auto& e : outs) {
      uint16_t tm = 0, dt = 0;
      dos_time(tm, dt);
      wr32(cd, kCdSig);
      wr16(cd, 20);        // version made by
      wr16(cd, 20);        // version needed
      wr16(cd, kFlagUtf8);
      wr16(cd, 8);
      wr16(cd, tm);
      wr16(cd, dt);
      wr32(cd, e.crc);
      wr32(cd, e.comp_size);
      wr32(cd, e.uncomp_size);
      wr16(cd, (uint16_t)e.name.size());
      wr16(cd, 0);         // extra len
      wr16(cd, 0);         // comment len
      wr16(cd, 0);         // disk number
      wr16(cd, 0);         // internal attrs
      wr32(cd, 0);         // external attrs
      wr32(cd, e.offset);
      cd += e.name;
    }
    os.write(cd.data(), (std::streamsize)cd.size());

    std::string eocd;
    wr32(eocd, kEocdSig);
    wr16(eocd, 0);                      // disk
    wr16(eocd, 0);                      // cd disk
    wr16(eocd, (uint16_t)outs.size());
    wr16(eocd, (uint16_t)outs.size());
    wr32(eocd, (uint32_t)cd.size());
    wr32(eocd, cd_offset);
    wr16(eocd, 0);                      // comment len
    os.write(eocd.data(), (std::streamsize)eocd.size());
    os.close();
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

} // namespace gpt
