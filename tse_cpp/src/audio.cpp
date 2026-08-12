#include "audio.h"

#include <alsa/asoundlib.h>

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace tse {

namespace {
snd_pcm_t* H(void* p) { return static_cast<snd_pcm_t*>(p); }
}  // namespace

AlsaDevice::AlsaDevice(const std::string& device, int sample_rate, int period,
                       bool capture)
    : capture_(capture) {
  snd_pcm_t* pcm = nullptr;

  const int err = snd_pcm_open(
      &pcm, device.c_str(),
      capture ? SND_PCM_STREAM_CAPTURE : SND_PCM_STREAM_PLAYBACK, 0);

  if (err < 0) {
    throw std::runtime_error(
        std::string(capture ? "Khong mo duoc mic '" : "Khong mo duoc loa '") +
        device + "': " + snd_strerror(err) +
        "\n  Xem thiet bi: ./tse_voice_lock devices");
  }

  handle_ = pcm;

  // Buffer = 4 x period: nhan doc/ghi van chay khi CPU dang ban chay model.
  const unsigned latency_us =
      static_cast<unsigned>(4.0 * period / sample_rate * 1e6);

  const int e2 = snd_pcm_set_params(
      pcm, SND_PCM_FORMAT_FLOAT_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
      1 /* mono */, static_cast<unsigned>(sample_rate),
      1 /* cho phep resample */, latency_us);

  if (e2 < 0) {
    snd_pcm_close(pcm);
    handle_ = nullptr;
    throw std::runtime_error(
        std::string("Thiet bi khong nhan cau hinh 16 kHz mono float32: ") +
        snd_strerror(e2) + "\n  Thu dung 'plughw:...' de ALSA tu convert.");
  }
}

AlsaDevice::~AlsaDevice() {
  if (handle_) {
    snd_pcm_drain(H(handle_));
    snd_pcm_close(H(handle_));
  }
}

bool AlsaDevice::Read(float* dst, int frames) {
  int done = 0;

  while (done < frames) {
    const snd_pcm_sframes_t n =
        snd_pcm_readi(H(handle_), dst + done, frames - done);

    if (n < 0) {
      ++xruns_;
      if (snd_pcm_recover(H(handle_), static_cast<int>(n), 1) < 0) return false;
      continue;
    }
    done += static_cast<int>(n);
  }

  return true;
}

bool AlsaDevice::Write(const float* src, int frames) {
  int done = 0;

  while (done < frames) {
    const snd_pcm_sframes_t n =
        snd_pcm_writei(H(handle_), src + done, frames - done);

    if (n < 0) {
      ++xruns_;
      if (snd_pcm_recover(H(handle_), static_cast<int>(n), 1) < 0) return false;
      continue;
    }
    done += static_cast<int>(n);
  }

  return true;
}

void AlsaDevice::Prefill(int frames) {
  const std::vector<float> silence(frames, 0.0f);
  Write(silence.data(), frames);
}

void ListDevices() {
  std::printf("Thiet bi PCM:\n\n");

  void** hints = nullptr;
  if (snd_device_name_hint(-1, "pcm", &hints) < 0) {
    std::printf("  (khong doc duoc danh sach)\n");
    return;
  }

  for (void** h = hints; *h; ++h) {
    char* name = snd_device_name_get_hint(*h, "NAME");
    char* desc = snd_device_name_get_hint(*h, "DESC");
    char* ioid = snd_device_name_get_hint(*h, "IOID");

    if (name) {
      const char* dir = ioid ? ioid : "Input/Output";
      std::printf("  %-28s [%s]\n", name, dir);

      if (desc) {
        // DESC hay co nhieu dong, chi lay dong dau cho gon
        std::string d(desc);
        const size_t nl = d.find('\n');
        if (nl != std::string::npos) d = d.substr(0, nl);
        std::printf("  %-28s  %s\n", "", d.c_str());
      }
    }

    free(name);
    free(desc);
    free(ioid);
  }

  snd_device_name_free_hint(hints);

  std::printf("\n  Dung ten o tren cho --in-device / --out-device.\n");
  std::printf("  Vi du: --in-device plughw:2,0\n");
}

}  // namespace tse
