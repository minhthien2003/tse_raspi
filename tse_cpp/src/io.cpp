#include "io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tse {

namespace {

uint32_t ReadU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t ReadU16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Khong mo duoc file: " + path);

  const std::streamsize n = f.tellg();
  f.seekg(0);

  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (n > 0 && !f.read(reinterpret_cast<char*>(buf.data()), n)) {
    throw std::runtime_error("Doc file loi: " + path);
  }
  return buf;
}

}  // namespace

// ===========================================================================
// WAV
// ===========================================================================

std::vector<float> ReadWav(const std::string& path, int* sample_rate) {
  const std::vector<uint8_t> b = ReadFile(path);

  if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) != 0 ||
      std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("Khong phai file WAV: " + path);
  }

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  size_t data_off = 0, data_len = 0;

  // Duyet chunk — khong gia dinh "fmt " va "data" nam ngay dau, vi nhieu
  // file co chunk LIST/fact chen vao giua.
  size_t pos = 12;
  while (pos + 8 <= b.size()) {
    const char* id = reinterpret_cast<const char*>(b.data() + pos);
    const uint32_t size = ReadU32(b.data() + pos + 4);
    const size_t body = pos + 8;

    if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16 &&
        body + 16 <= b.size()) {
      format   = ReadU16(b.data() + body);
      channels = ReadU16(b.data() + body + 2);
      rate     = ReadU32(b.data() + body + 4);
      bits     = ReadU16(b.data() + body + 14);
    } else if (std::memcmp(id, "data", 4) == 0) {
      data_off = body;
      data_len = std::min(static_cast<size_t>(size), b.size() - body);
    }

    pos = body + size + (size & 1);   // chunk luon can le chan
  }

  if (data_off == 0 || channels == 0) {
    throw std::runtime_error("WAV thieu chunk fmt/data: " + path);
  }
  // 1 = PCM, 3 = IEEE float, 0xFFFE = extensible
  if (format != 1 && format != 3 && format != 0xFFFE) {
    throw std::runtime_error("WAV dung codec nen, chua ho tro: " + path);
  }

  const int bytes = bits / 8;
  if (bytes < 2 || bytes > 4) {
    throw std::runtime_error("WAV chi ho tro 16/24/32-bit, file nay " +
                             std::to_string(bits) + "-bit");
  }

  const size_t frame_bytes = static_cast<size_t>(bytes) * channels;
  const size_t frames = frame_bytes ? data_len / frame_bytes : 0;

  std::vector<float> out(frames);

  for (size_t i = 0; i < frames; ++i) {
    double acc = 0.0;

    for (int c = 0; c < channels; ++c) {
      const uint8_t* p =
          b.data() + data_off + (i * channels + c) * bytes;

      if (format == 3 && bytes == 4) {
        float v;
        std::memcpy(&v, p, 4);
        acc += v;
      } else if (bytes == 2) {
        acc += static_cast<int16_t>(ReadU16(p)) / 32768.0;
      } else if (bytes == 3) {
        int32_t v = (p[0] << 8) | (p[1] << 16) | (p[2] << 24);
        acc += (v >> 8) / 8388608.0;
      } else {
        acc += static_cast<int32_t>(ReadU32(p)) / 2147483648.0;
      }
    }

    out[i] = static_cast<float>(acc / channels);
  }

  if (sample_rate) *sample_rate = static_cast<int>(rate);
  return out;
}

void WriteWav(const std::string& path, const std::vector<float>& samples,
              int sample_rate) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Khong ghi duoc file: " + path);

  const uint32_t n = static_cast<uint32_t>(samples.size());
  const uint32_t data_bytes = n * 2;
  const uint32_t rate = static_cast<uint32_t>(sample_rate);

  auto u32 = [&f](uint32_t v) {
    uint8_t p[4] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
                    static_cast<uint8_t>(v >> 16),
                    static_cast<uint8_t>(v >> 24)};
    f.write(reinterpret_cast<char*>(p), 4);
  };
  auto u16 = [&f](uint16_t v) {
    uint8_t p[2] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8)};
    f.write(reinterpret_cast<char*>(p), 2);
  };

  f.write("RIFF", 4);
  u32(36 + data_bytes);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  u32(16);
  u16(1);            // PCM
  u16(1);            // mono
  u32(rate);
  u32(rate * 2);     // byte rate
  u16(2);            // block align
  u16(16);           // bits
  f.write("data", 4);
  u32(data_bytes);

  for (float v : samples) {
    const float c = std::max(-1.0f, std::min(1.0f, v));
    u16(static_cast<uint16_t>(static_cast<int16_t>(std::lrint(c * 32767.0f))));
  }
}

// ===========================================================================
// NPY
// ===========================================================================

std::vector<float> ReadNpyFloat32(const std::string& path) {
  const std::vector<uint8_t> b = ReadFile(path);

  if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Khong phai file .npy: " + path);
  }

  const uint8_t major = b[6];
  size_t header_len, header_off;

  if (major == 1) {
    header_len = ReadU16(b.data() + 8);
    header_off = 10;
  } else {
    header_len = ReadU32(b.data() + 8);
    header_off = 12;
  }

  if (header_off + header_len > b.size()) {
    throw std::runtime_error("Header .npy hong: " + path);
  }

  const std::string header(reinterpret_cast<const char*>(b.data() + header_off),
                           header_len);

  if (header.find("'<f4'") == std::string::npos &&
      header.find("\"<f4\"") == std::string::npos) {
    throw std::runtime_error(
        "File .npy phai la float32 ('<f4'). Sua o Python:\n"
        "    np.save(path, emb.astype(np.float32))\n"
        "  header: " + header);
  }
  if (header.find("'fortran_order': True") != std::string::npos) {
    throw std::runtime_error("Khong ho tro .npy fortran_order: " + path);
  }

  const size_t data_off = header_off + header_len;
  const size_t count = (b.size() - data_off) / 4;

  std::vector<float> out(count);
  if (count) std::memcpy(out.data(), b.data() + data_off, count * 4);

  return out;
}

void WriteNpyFloat32(const std::string& path, const std::vector<float>& data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Khong ghi duoc file: " + path);

  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                     std::to_string(data.size()) + ",), }";

  // Numpy yeu cau (10 + len(dict)) chia het cho 64, ket thuc bang '\n'.
  size_t total = 10 + dict.size() + 1;
  const size_t pad = (64 - (total % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');

  const uint16_t hlen = static_cast<uint16_t>(dict.size());

  f.write("\x93NUMPY", 6);
  const uint8_t ver[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(ver), 2);

  const uint8_t hl[2] = {static_cast<uint8_t>(hlen & 0xFF),
                         static_cast<uint8_t>(hlen >> 8)};
  f.write(reinterpret_cast<const char*>(hl), 2);

  f.write(dict.data(), static_cast<std::streamsize>(dict.size()));
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size() * 4));
}

}  // namespace tse
