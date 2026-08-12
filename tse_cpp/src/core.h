// core.h — FFT + STFT/iSTFT + tien ich DSP cho V49 voice lock.
//
// Khong phu thuoc thu vien ngoai. Moi hang so phai khop chinh xac voi
// luc training/export cua V49, neu lech thi model chay ra rac.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace tse {

// ---------------------------------------------------------------------------
// Cau hinh — KHOP VOI V49 ONNX EXPORT, khong duoc sua tuy tien
// ---------------------------------------------------------------------------

constexpr int   kSampleRate = 16000;
constexpr int   kNFft       = 512;
constexpr int   kHop        = 128;
constexpr int   kNFreq      = kNFft / 2 + 1;   // 257
constexpr float kComp       = 0.3f;
constexpr float kEps        = 1e-8f;
constexpr int   kSpkDim     = 512;

// Enrollment: segment 3 s, chong lan 1.5 s.
constexpr int kEnrollSegment = 3 * kSampleRate;
constexpr int kEnrollHop     = kSampleRate * 3 / 2;

// Nguong mac dinh
constexpr float kDefaultLockDb = -8.0f;
constexpr float kSilenceRms    = 3e-3f;

// ---------------------------------------------------------------------------
// FFT phuc, radix-2 Cooley-Tukey, tai cho
// ---------------------------------------------------------------------------

class Fft {
 public:
  // n phai la luy thua cua 2.
  explicit Fft(int n);

  void Forward(std::vector<std::complex<float>>& a) const;
  // Da chia cho n.
  void Inverse(std::vector<std::complex<float>>& a) const;

  int size() const { return n_; }

 private:
  void Transform(std::vector<std::complex<float>>& a, bool inverse) const;

  int                                n_;
  std::vector<int>                   rev_;
  std::vector<std::complex<float>>   tw_;   // exp(-2*pi*i*j/n), j < n/2
};

// ---------------------------------------------------------------------------
// Ket qua STFT. Cac mang luu row-major [freq][frame] -> index f*frames + t,
// dung y het layout numpy [F, T] ma ONNX mong doi.
// ---------------------------------------------------------------------------

struct Spectrogram {
  int                frames = 0;
  std::vector<float> real;   // kNFreq * frames
  std::vector<float> imag;
  std::vector<float> mag;
};

// Cua so Hann PERIODIC: 0.5 - 0.5*cos(2*pi*n/N).
// KHONG phai symmetric (np.hanning) — training dung torch.hann_window.
std::vector<float> HannPeriodic(int n);

class Stft {
 public:
  Stft();

  // center=True, pad reflect kNFft/2 moi ben — giong torch.stft.
  Spectrogram Forward(const float* audio, int len) const;

  // Dung lai pho day du tu kNFreq bin (doi xung Hermitian), overlap-add,
  // chia cho tong binh phuong cua so, roi cat bo phan pad.
  void Inverse(const float* real, const float* imag, int frames,
               float* out, int out_len) const;

  int FramesFor(int len) const;

 private:
  Fft                fft_;
  std::vector<float> win_;
  std::vector<float> win_sq_;
};

// ---------------------------------------------------------------------------
// Tien ich
// ---------------------------------------------------------------------------

// comp = (mag+eps)^0.3 ; out = (comp - mean) / max(std, 1e-3)
void NormalizeInput(const std::vector<float>& mag, std::vector<float>& out);

float Rms(const float* x, int n);
float Db(float x);

// Chuan hoa mean/var cho WavLM (giong Wav2Vec2FeatureExtractor).
void NormalizeMeanVar(const float* in, int n, std::vector<float>& out);

}  // namespace tse
