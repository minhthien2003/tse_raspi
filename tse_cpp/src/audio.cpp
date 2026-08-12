#include "audio.h"

#include <alsa/asoundlib.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace tse {

namespace {

snd_pcm_t* H(void* p) { return static_cast<snd_pcm_t*>(p); }

// ALSA in thang loi ra stderr. Khi dang do thiet bi thi tat di cho do roi.
void SilentHandler(const char*, int, const char*, int, const char*, ...) {}

struct QuietAlsa {
  QuietAlsa()  { snd_lib_error_set_handler(SilentHandler); }
  ~QuietAlsa() { snd_lib_error_set_handler(nullptr); }
};

// Mo PCM, tra ve nullptr neu that bai (khong nem exception).
snd_pcm_t* TryOpen(const std::string& name, int sample_rate, int period,
                   bool capture) {
  snd_pcm_t* pcm = nullptr;

  if (snd_pcm_open(&pcm, name.c_str(),
                   capture ? SND_PCM_STREAM_CAPTURE : SND_PCM_STREAM_PLAYBACK,
                   0) < 0) {
    return nullptr;
  }

  const unsigned latency_us =
      static_cast<unsigned>(4.0 * period / sample_rate * 1e6);

  if (snd_pcm_set_params(pcm, SND_PCM_FORMAT_FLOAT_LE,
                         SND_PCM_ACCESS_RW_INTERLEAVED, 1,
                         static_cast<unsigned>(sample_rate), 1,
                         latency_us) < 0) {
    snd_pcm_close(pcm);
    return nullptr;
  }

  return pcm;
}

// Duyet cac card tim thiet bi dung duoc. Uu tien plughw: de ALSA tu
// convert sample rate/format — nhieu mic USB khong chay 16 kHz native.
std::string FindWorkingDevice(int sample_rate, int period, bool capture) {
  QuietAlsa quiet;

  int card = -1;
  while (snd_card_next(&card) == 0 && card >= 0) {
    for (const char* prefix : {"plughw:", "hw:"}) {
      const std::string name = std::string(prefix) + std::to_string(card) + ",0";

      if (snd_pcm_t* pcm = TryOpen(name, sample_rate, period, capture)) {
        snd_pcm_close(pcm);
        return name;
      }
    }
  }

  return "";
}

}  // namespace

AlsaDevice::AlsaDevice(const std::string& device, int sample_rate, int period,
                       bool capture)
    : capture_(capture) {
  const char* what = capture ? "mic" : "loa";

  // Buffer = 4 x period: nhan doc/ghi van chay khi CPU dang ban chay model.
  snd_pcm_t* pcm = nullptr;
  {
    QuietAlsa quiet;   // giau loi ALSA o lan thu dau, ta se tu bao sau
    pcm = TryOpen(device, sample_rate, period, capture);
  }

  if (pcm) {
    handle_ = pcm;
    return;
  }

  // 'default' hay hong tren Pi khi asound.conf dinh nghia kieu asym ma
  // thieu nhanh capture. Tu do card that thay vi bo cuoc.
  const std::string fallback = FindWorkingDevice(sample_rate, period, capture);

  if (fallback.empty()) {
    throw std::runtime_error(
        std::string("Khong mo duoc ") + what + " '" + device + "', va khong "
        "tim thay thiet bi thay the nao.\n"
        "  Xem danh sach : ./tse_voice_lock devices\n"
        "                  " + (capture ? "arecord -l" : "aplay -l") + "\n"
        "  Chi dinh tay  : --" + (capture ? "in" : "out") +
        "-device plughw:<card>,0\n"
        "\n"
        "  Neu loi la 'capture slave is not defined' thi ~/.asoundrc hoac\n"
        "  /etc/asound.conf dang khai bao 'default' kieu asym ma thieu\n"
        "  nhanh capture. Sua lai cho co ca hai:\n"
        "      pcm.!default {\n"
        "          type asym\n"
        "          playback.pcm \"plughw:0,0\"\n"
        "          capture.pcm  \"plughw:1,0\"\n"
        "      }");
  }

  std::printf("  '%s' khong mo duoc cho %s -> dung '%s'\n",
              device.c_str(), what, fallback.c_str());

  pcm = TryOpen(fallback, sample_rate, period, capture);

  if (!pcm) {
    throw std::runtime_error(std::string("Mat thiet bi '") + fallback +
                             "' giua chung");
  }

  handle_ = pcm;
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
  // Card that — day moi la cai can de go plughw:<card>,0
  std::printf("Card am thanh:\n\n");

  int card = -1;
  bool any = false;

  while (snd_card_next(&card) == 0 && card >= 0) {
    char* name = nullptr;
    snd_card_get_name(card, &name);

    std::printf("  card %d: %s\n", card, name ? name : "?");
    std::printf("           plughw:%d,0", card);

    // Kiem tra thuc te card nay thu / phat duoc khong
    {
      QuietAlsa quiet;
      const std::string dev = "plughw:" + std::to_string(card) + ",0";

      std::string caps;
      if (snd_pcm_t* p = TryOpen(dev, 16000, 4096, true)) {
        snd_pcm_close(p);
        caps += " THU";
      }
      if (snd_pcm_t* p = TryOpen(dev, 16000, 4096, false)) {
        snd_pcm_close(p);
        caps += " PHAT";
      }

      std::printf("%s\n", caps.empty() ? "  (khong dung duoc o 16 kHz mono)"
                                       : ("  ->" + caps).c_str());
    }

    free(name);
    any = true;
  }

  if (!any) std::printf("  (khong thay card nao)\n");

  std::printf("\n  Vi du: --in-device plughw:1,0 --out-device plughw:0,0\n");

  std::printf("\n----------------------------------------------------------\n");
  std::printf("Ten PCM (bao gom ca plugin ao):\n\n");

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

  std::printf("\n  Neu 'default' bao 'capture slave is not defined' thi\n");
  std::printf("  bo qua no va dung plughw:<card>,0 o tren.\n");
}

}  // namespace tse
