#include "audio.h"

#include <alsa/asoundlib.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace tse {

namespace {

snd_pcm_t* H(void* p) { return static_cast<snd_pcm_t*>(p); }

// ALSA prints errors straight to stderr. Silence it while probing devices,
// otherwise the output is full of noise from the attempts that fail.
void SilentHandler(const char*, int, const char*, int, const char*, ...) {}

struct QuietAlsa {
  QuietAlsa()  { snd_lib_error_set_handler(SilentHandler); }
  ~QuietAlsa() { snd_lib_error_set_handler(nullptr); }
};

// Opens a PCM device, returning nullptr on failure (never throws).
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

// Walks the cards looking for one that works. plughw: is preferred so ALSA
// converts sample rate and format for us - many USB mics cannot do 16 kHz
// natively.
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
  const char* what = capture ? "microphone" : "speaker";

  // Buffer = 4 x period: reads and writes keep flowing while the CPU is
  // busy running the model.
  snd_pcm_t* pcm = nullptr;
  {
    QuietAlsa quiet;   // hide ALSA's own error on the first attempt
    pcm = TryOpen(device, sample_rate, period, capture);
  }

  if (pcm) {
    handle_ = pcm;
    return;
  }

  // 'default' is often broken on the Pi when asound.conf declares an asym
  // device without a capture branch. Probe real cards instead of giving up.
  const std::string fallback = FindWorkingDevice(sample_rate, period, capture);

  if (fallback.empty()) {
    throw std::runtime_error(
        std::string("Cannot open the ") + what + " '" + device + "', and no "
        "usable replacement device was found.\n"
        "  List devices  : ./tse_voice_lock devices\n"
        "                  " + (capture ? "arecord -l" : "aplay -l") + "\n"
        "  Set it by hand: --" + (capture ? "in" : "out") +
        "-device plughw:<card>,0\n"
        "\n"
        "  If the error says 'capture slave is not defined', then\n"
        "  ~/.asoundrc or /etc/asound.conf declares 'default' as an asym\n"
        "  device without a capture branch. Give it both:\n"
        "      pcm.!default {\n"
        "          type asym\n"
        "          playback.pcm \"plughw:0,0\"\n"
        "          capture.pcm  \"plughw:1,0\"\n"
        "      }");
  }

  std::printf("  '%s' would not open for %s -> using '%s'\n",
              device.c_str(), what, fallback.c_str());

  pcm = TryOpen(fallback, sample_rate, period, capture);

  if (!pcm) {
    throw std::runtime_error(std::string("Lost device '") + fallback +
                             "' midway");
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
  // Real cards - this is what you need in order to type plughw:<card>,0
  std::printf("Sound cards:\n\n");

  int card = -1;
  bool any = false;

  while (snd_card_next(&card) == 0 && card >= 0) {
    char* name = nullptr;
    snd_card_get_name(card, &name);

    std::printf("  card %d: %s\n", card, name ? name : "?");
    std::printf("           plughw:%d,0", card);

    // Actually try the card so the capabilities shown are real
    {
      QuietAlsa quiet;
      const std::string dev = "plughw:" + std::to_string(card) + ",0";

      std::string caps;
      if (snd_pcm_t* p = TryOpen(dev, 16000, 4096, true)) {
        snd_pcm_close(p);
        caps += " CAPTURE";
      }
      if (snd_pcm_t* p = TryOpen(dev, 16000, 4096, false)) {
        snd_pcm_close(p);
        caps += " PLAYBACK";
      }

      std::printf("%s\n", caps.empty() ? "  (unusable at 16 kHz mono)"
                                       : ("  ->" + caps).c_str());
    }

    free(name);
    any = true;
  }

  if (!any) std::printf("  (no cards found)\n");

  std::printf("\n  Example: --in-device plughw:1,0 --out-device plughw:0,0\n");

  std::printf("\n----------------------------------------------------------\n");
  std::printf("PCM names (including virtual plugins):\n\n");

  void** hints = nullptr;
  if (snd_device_name_hint(-1, "pcm", &hints) < 0) {
    std::printf("  (could not read the device list)\n");
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
        // DESC is often multi-line; keep the first line only
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

  std::printf("\n  If 'default' reports 'capture slave is not defined',\n");
  std::printf("  ignore it and use one of the plughw:<card>,0 names above.\n");
}

}  // namespace tse
