// core.h - FFT + STFT/iSTFT + DSP helpers for V49 voice lock.
//
// No external dependencies. Every constant here must match the values used
// during V49 training/export exactly; a mismatch makes the model emit noise
// without raising any error.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace tse {

// ---------------------------------------------------------------------------
// Configuration - MATCHES V49 ONNX EXPORT, do not change casually
// ---------------------------------------------------------------------------

constexpr int   kSampleRate = 16000;
constexpr int   kNFft       = 512;
constexpr int   kHop        = 128;
constexpr int   kNFreq      = kNFft / 2 + 1;   // 257
constexpr float kComp       = 0.3f;
constexpr float kEps        = 1e-8f;
constexpr int   kSpkDim     = 512;

// Enrollment: 3 s segments with 1.5 s overlap.
constexpr int kEnrollSegment = 3 * kSampleRate;
constexpr int kEnrollHop     = kSampleRate * 3 / 2;

// Defaults
constexpr float kDefaultLockDb = -8.0f;
constexpr float kSilenceRms    = 3e-3f;

// ---------------------------------------------------------------------------
// Complex radix-2 Cooley-Tukey FFT, in place
// ---------------------------------------------------------------------------

class Fft {
 public:
  // n must be a power of two.
  explicit Fft(int n);

  void Forward(std::vector<std::complex<float>>& a) const;
  // Already divided by n.
  void Inverse(std::vector<std::complex<float>>& a) const;

  int size() const { return n_; }

 private:
  void Transform(std::vector<std::complex<float>>& a, bool inverse) const;

  int                                n_;
  std::vector<int>                   rev_;
  std::vector<std::complex<float>>   tw_;   // exp(-2*pi*i*j/n), j < n/2
};

// ---------------------------------------------------------------------------
// STFT result. Arrays are row-major [freq][frame] -> index f*frames + t,
// which is exactly the numpy [F, T] layout the ONNX model expects.
// ---------------------------------------------------------------------------

struct Spectrogram {
  int                frames = 0;
  std::vector<float> real;   // kNFreq * frames
  std::vector<float> imag;
  std::vector<float> mag;
};

// PERIODIC Hann window: 0.5 - 0.5*cos(2*pi*n/N).
// NOT the symmetric variant (np.hanning) - training used torch.hann_window.
std::vector<float> HannPeriodic(int n);

class Stft {
 public:
  Stft();

  // center=True with reflect padding of kNFft/2 on each side, like torch.stft.
  Spectrogram Forward(const float* audio, int len) const;

  // Rebuilds the full spectrum from kNFreq bins via Hermitian symmetry,
  // overlap-adds, divides by the summed squared window, then strips padding.
  void Inverse(const float* real, const float* imag, int frames,
               float* out, int out_len) const;

  int FramesFor(int len) const;

 private:
  Fft                fft_;
  std::vector<float> win_;
  std::vector<float> win_sq_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// comp = (mag+eps)^0.3 ; out = (comp - mean) / max(std, 1e-3)
void NormalizeInput(const std::vector<float>& mag, std::vector<float>& out);

float Rms(const float* x, int n);
float Db(float x);

// Mean/variance normalization for WavLM (matches Wav2Vec2FeatureExtractor).
void NormalizeMeanVar(const float* in, int n, std::vector<float>& out);

}  // namespace tse
