// io.h — doc/ghi WAV va NPY, khong dung thu vien ngoai.
//
// NPY phai tuong thich hai chieu voi numpy: file speaker_emb.npy tao tren
// laptop bang np.save phai doc duoc o day, va nguoc lai.
#pragma once

#include <string>
#include <vector>

namespace tse {

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

// Doc WAV mono/stereo, PCM 16/24/32-bit hoac float32. Stereo bi tron thanh
// mono. Tra ve mau o [-1, 1]. Nem std::runtime_error neu loi.
// sample_rate nhan lai tan so that su trong file (KHONG resample).
std::vector<float> ReadWav(const std::string& path, int* sample_rate);

// Ghi WAV mono PCM 16-bit.
void WriteWav(const std::string& path, const std::vector<float>& samples,
              int sample_rate);

// ---------------------------------------------------------------------------
// NPY (float32, 1-D hoac 2-D, C-order)
// ---------------------------------------------------------------------------

std::vector<float> ReadNpyFloat32(const std::string& path);

void WriteNpyFloat32(const std::string& path, const std::vector<float>& data);

}  // namespace tse
