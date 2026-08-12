// audio.h — thu/phat qua ALSA. Khong dung PortAudio de giam phu thuoc
// khi dong image nhung (Yocto / i.MX 95).
#pragma once

#include <string>
#include <vector>

namespace tse {

class AlsaDevice {
 public:
  // `device` kieu "default", "plughw:1,0". `period` la so frame moi lan
  // doc/ghi. Buffer duoc dat 4 x period de nhan doc van chay tiep trong
  // luc CPU dang chay model.
  AlsaDevice(const std::string& device, int sample_rate, int period,
             bool capture);
  ~AlsaDevice();

  AlsaDevice(const AlsaDevice&) = delete;
  AlsaDevice& operator=(const AlsaDevice&) = delete;

  // Tra ve false neu khong khoi phuc duoc sau xrun.
  bool Read(float* dst, int frames);
  bool Write(const float* src, int frames);

  // Ghi im lang de mo buffer phat truoc khi vao vong lap.
  void Prefill(int frames);

  int xruns() const { return xruns_; }

 private:
  void* handle_ = nullptr;   // snd_pcm_t*
  bool  capture_;
  int   xruns_ = 0;
};

// In danh sach card/thiet bi PCM.
void ListDevices();

}  // namespace tse
