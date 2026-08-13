#include "core.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tse {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

// ===========================================================================
// Fft
// ===========================================================================

Fft::Fft(int n) : n_(n) {
  if (n <= 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("Fft: size must be a power of two");
  }

  // Bit-reversal table
  rev_.resize(n);
  int bits = 0;
  while ((1 << bits) < n) ++bits;

  for (int i = 0; i < n; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) {
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    }
    rev_[i] = r;
  }

  // Twiddles exp(-2*pi*i*j/n) - computed in double, stored as float to
  // keep the rounding error down.
  tw_.resize(n / 2);
  for (int j = 0; j < n / 2; ++j) {
    double a = -2.0 * kPi * j / n;
    tw_[j] = std::complex<float>(static_cast<float>(std::cos(a)),
                                 static_cast<float>(std::sin(a)));
  }
}

void Fft::Transform(std::vector<std::complex<float>>& a, bool inverse) const {
  if (static_cast<int>(a.size()) != n_) {
    throw std::invalid_argument("Fft: input length mismatch");
  }

  for (int i = 0; i < n_; ++i) {
    int j = rev_[i];
    if (i < j) std::swap(a[i], a[j]);
  }

  for (int len = 2; len <= n_; len <<= 1) {
    const int half = len >> 1;
    const int step = n_ / len;

    for (int i = 0; i < n_; i += len) {
      for (int k = 0; k < half; ++k) {
        std::complex<float> w = tw_[k * step];
        if (inverse) w = std::conj(w);

        const std::complex<float> u = a[i + k];
        const std::complex<float> v = a[i + k + half] * w;

        a[i + k]        = u + v;
        a[i + k + half] = u - v;
      }
    }
  }
}

void Fft::Forward(std::vector<std::complex<float>>& a) const {
  Transform(a, false);
}

void Fft::Inverse(std::vector<std::complex<float>>& a) const {
  Transform(a, true);
  const float inv = 1.0f / static_cast<float>(n_);
  for (auto& x : a) x *= inv;
}

// ===========================================================================
// Window
// ===========================================================================

std::vector<float> HannPeriodic(int n) {
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i) {
    w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / n));
  }
  return w;
}

// ===========================================================================
// Stft
// ===========================================================================

Stft::Stft() : fft_(kNFft), win_(HannPeriodic(kNFft)), win_sq_(kNFft) {
  for (int i = 0; i < kNFft; ++i) win_sq_[i] = win_[i] * win_[i];
}

int Stft::FramesFor(int len) const {
  const int padded = len + kNFft;          // kNFft/2 padding on each side
  return 1 + (padded - kNFft) / kHop;
}

namespace {

// np.pad(x, (p, p), mode="reflect"): padded[i] = x[p-i] for i<p, and
// padded[p+n+j] = x[n-2-j] for j<p. Requires n >= p+1.
std::vector<float> ReflectPad(const float* x, int n, int p) {
  std::vector<float> out(static_cast<size_t>(n) + 2 * p);

  for (int i = 0; i < p; ++i) {
    int src = p - i;
    if (src >= n) src = n > 0 ? n - 1 : 0;      // unusually short audio
    out[i] = x[src];
  }

  std::copy(x, x + n, out.begin() + p);

  for (int j = 0; j < p; ++j) {
    int src = n - 2 - j;
    if (src < 0) src = 0;
    out[static_cast<size_t>(p) + n + j] = x[src];
  }

  return out;
}

}  // namespace

Spectrogram Stft::Forward(const float* audio, int len) const {
  const int pad = kNFft / 2;
  const std::vector<float> x = ReflectPad(audio, len, pad);

  Spectrogram s;
  s.frames = 1 + (static_cast<int>(x.size()) - kNFft) / kHop;

  const size_t total = static_cast<size_t>(kNFreq) * s.frames;
  s.real.resize(total);
  s.imag.resize(total);
  s.mag.resize(total);

  std::vector<std::complex<float>> buf(kNFft);

  for (int t = 0; t < s.frames; ++t) {
    const float* seg = x.data() + static_cast<size_t>(t) * kHop;

    for (int i = 0; i < kNFft; ++i) {
      buf[i] = std::complex<float>(seg[i] * win_[i], 0.0f);
    }

    fft_.Forward(buf);

    for (int f = 0; f < kNFreq; ++f) {
      const size_t idx = static_cast<size_t>(f) * s.frames + t;
      const float re = buf[f].real();
      const float im = buf[f].imag();

      s.real[idx] = re;
      s.imag[idx] = im;
      s.mag[idx]  = std::sqrt(re * re + im * im + kEps);
    }
  }

  return s;
}

void Stft::Inverse(const float* real, const float* imag, int frames,
                   float* out, int out_len) const {
  const int pad = kNFft / 2;
  const size_t full_len =
      static_cast<size_t>(frames - 1) * kHop + kNFft;

  std::vector<float> acc(full_len, 0.0f);
  std::vector<float> wsum(full_len, 0.0f);
  std::vector<std::complex<float>> buf(kNFft);

  for (int t = 0; t < frames; ++t) {
    // Lower half comes from the model
    for (int f = 0; f < kNFreq; ++f) {
      const size_t idx = static_cast<size_t>(f) * frames + t;
      buf[f] = std::complex<float>(real[idx], imag[idx]);
    }
    // Upper half follows from Hermitian symmetry: X[N-k] = conj(X[k])
    for (int f = kNFreq; f < kNFft; ++f) {
      buf[f] = std::conj(buf[kNFft - f]);
    }

    fft_.Inverse(buf);

    const size_t start = static_cast<size_t>(t) * kHop;
    for (int i = 0; i < kNFft; ++i) {
      acc[start + i]  += buf[i].real() * win_[i];
      wsum[start + i] += win_sq_[i];
    }
  }

  for (int i = 0; i < out_len; ++i) {
    const size_t idx = static_cast<size_t>(pad) + i;
    out[i] = idx < full_len
                 ? acc[idx] / std::max(wsum[idx], 1e-8f)
                 : 0.0f;
  }
}

// ===========================================================================
// Helpers
// ===========================================================================

void NormalizeInput(const std::vector<float>& mag, std::vector<float>& out) {
  const size_t n = mag.size();
  out.resize(n);

  // Accumulate in double: with 96632 elements a float32 sum drifts enough
  // to shift the mean and std.
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    out[i] = std::pow(mag[i] + kEps, kComp);
    sum += out[i];
  }

  const double mean = sum / static_cast<double>(n);

  double var = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = out[i] - mean;
    var += d * d;
  }

  const double stddev = std::sqrt(var / static_cast<double>(n));
  const float inv = 1.0f / static_cast<float>(std::max(stddev, 1e-3));
  const float m = static_cast<float>(mean);

  for (size_t i = 0; i < n; ++i) out[i] = (out[i] - m) * inv;
}

float Rms(const float* x, int n) {
  if (n <= 0) return 0.0f;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
  return static_cast<float>(std::sqrt(sum / n + 1e-20));
}

float Db(float x) {
  return 20.0f * std::log10(std::max(x, 1e-8f));
}

void NormalizeMeanVar(const float* in, int n, std::vector<float>& out) {
  out.resize(n);

  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += in[i];
  const double mean = sum / n;

  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = in[i] - mean;
    var += d * d;
  }
  // ddof=0 - matches np.var in the Python port and HuggingFace's
  // Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm.
  var /= n;

  const float inv = static_cast<float>(1.0 / std::sqrt(var + 1e-7));
  const float m = static_cast<float>(mean);

  for (int i = 0; i < n; ++i) out[i] = (in[i] - m) * inv;
}

}  // namespace tse
