// audio.h - capture and playback through ALSA. PortAudio is deliberately
// avoided to keep the dependency list short when building an embedded image
// (Yocto / i.MX 95).
#pragma once

#include <string>
#include <vector>

namespace tse {

class AlsaDevice {
 public:
  // `device` looks like "default" or "plughw:1,0". `period` is the number of
  // frames per read/write. The buffer is sized at 4 x period so the kernel
  // keeps capturing while the CPU is busy running the model.
  AlsaDevice(const std::string& device, int sample_rate, int period,
             bool capture);
  ~AlsaDevice();

  AlsaDevice(const AlsaDevice&) = delete;
  AlsaDevice& operator=(const AlsaDevice&) = delete;

  // Returns false when recovery from an xrun fails.
  bool Read(float* dst, int frames);
  bool Write(const float* src, int frames);

  // Writes silence to prime the playback buffer before entering the loop.
  void Prefill(int frames);

  int xruns() const { return xruns_; }

 private:
  void* handle_ = nullptr;   // snd_pcm_t*
  bool  capture_;
  int   xruns_ = 0;
};

// Prints the available sound cards and PCM devices.
void ListDevices();

}  // namespace tse
