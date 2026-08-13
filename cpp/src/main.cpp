// main.cpp - CLI for the V49 voice lock, C++ build.
//
// The modes keep the same names and meanings as the Python port
// (voice_lock.py) so the two can be compared directly against each other.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "audio.h"
#include "core.h"
#include "enroll.h"
#include "io.h"
#include "voice_lock.h"

using namespace tse;

namespace {

std::atomic<bool> g_stop{false};

void OnSigint(int) { g_stop = true; }

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
  std::string cmd;
  Options     opt;

  std::string input, output = "extracted.wav";
  std::string emb_target, emb_other;
  std::string encoder, save;
  std::string in_device = "default", out_device = "default";
  std::string out_wav = "enrollment.wav";

  float seconds = 10.0f;
  int   iters   = 20;

  std::string config_used;   // config file that was loaded, for reporting
};

[[noreturn]] void Die(const std::string& msg) {
  std::fprintf(stderr, "\nERROR: %s\n", msg.c_str());
  std::exit(1);
}

// ---------------------------------------------------------------------------
// Config file
//
// Format is key = value with '#' comments. Applied BEFORE argv is read, so
// command line arguments always win over the file.
// ---------------------------------------------------------------------------

const char* kConfigKeys[] = {
    "model", "emb", "encoder", "power", "gain", "lock_db", "threads",
    "in_device", "out_device", "save", "input", "output", "out_wav",
    "seconds", "iters", "norm",
};

void ApplyOption(const std::string& key, const std::string& value, Args& a,
                 const char* origin) {
  if      (key == "model")      a.opt.model   = value;
  else if (key == "emb")        a.opt.emb     = value;
  else if (key == "encoder")    a.encoder     = value;
  else if (key == "power")      a.opt.power   = std::atof(value.c_str());
  else if (key == "gain")       a.opt.gain_db = std::atof(value.c_str());
  else if (key == "lock_db")    a.opt.lock_db = std::atof(value.c_str());
  else if (key == "threads")    a.opt.threads = std::atoi(value.c_str());
  else if (key == "in_device")  a.in_device   = value;
  else if (key == "out_device") a.out_device  = value;
  else if (key == "save")       a.save        = value;
  else if (key == "input")      a.input       = value;
  else if (key == "output")     a.output      = value;
  else if (key == "out_wav")    a.out_wav     = value;
  else if (key == "seconds")    a.seconds     = std::atof(value.c_str());
  else if (key == "iters")      a.iters       = std::atoi(value.c_str());
  else if (key == "norm") {
    if (value != "off" && value != "peak") {
      Die(std::string(origin) + ": norm accepts only 'off' or 'peak', got '" +
          value + "'");
    }
    a.opt.norm_peak = value == "peak";
  }
  else {
    // Silently ignoring a misspelled key is a trap: you think you changed
    // the configuration when nothing actually changed.
    std::string valid;
    for (const char* k : kConfigKeys) valid += std::string("  ") + k + "\n";

    Die(std::string("Invalid key in ") + origin + ": '" + key +
        "'\n  Valid keys:\n" + valid);
  }
}

std::string Trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// Returns the path that was used, or an empty string when no file was found.
std::string LoadConfig(const std::string& path, Args& a) {
  const std::string found = ResolvePath(path);
  if (found.empty()) return "";

  std::ifstream f(found);
  if (!f) return "";

  std::string line;
  int lineno = 0;

  while (std::getline(f, line)) {
    ++lineno;

    const size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);

    line = Trim(line);
    if (line.empty()) continue;

    const size_t eq = line.find('=');
    if (eq == std::string::npos) {
      Die(found + ":" + std::to_string(lineno) +
          ": missing '=' on line: " + line);
    }

    std::string key = Trim(line.substr(0, eq));
    const std::string value = Trim(line.substr(eq + 1));

    // Accept both 'lock-db' and 'lock_db'
    std::replace(key.begin(), key.end(), '-', '_');

    if (value.empty()) continue;   // key present but blank = keep the default

    ApplyOption(key, value, a,
                (found + ":" + std::to_string(lineno)).c_str());
  }

  return found;
}

void Usage() {
  std::printf(R"(V49 VOICE LOCK - C++ build for Raspberry Pi 5

  tse_voice_lock <mode> [options]

Modes:
  enroll    build a speaker embedding from the mic or a wav file
  bench     measure latency / RTF
  file      extract the target voice from a wav file
  verify    A/B the correct voice against a different one
  live      realtime through the mic (WEAR HEADPHONES)
  devices   list ALSA devices
  selftest  check STFT/iSTFT, no model required

Common options:
  --config PATH     config file (voice_lock.conf is found automatically)
  --no-config       skip the config file, use defaults plus command line
  --model PATH      defaults to v49_int8.onnx (searched under models/)
  --emb PATH        defaults to speaker_emb.npy
  --power N         mask sharpening (1=off, 2=default, 3=strong)
  --gain N          output gain in dB
  --lock-db N       LOCKED threshold (default -8)
  --threads N       ONNX thread count (default 4)
  --norm off|peak   normalize the OUTPUT FILE to peak 0.95. Applied after
                    measurement, so it does not skew the suppression figures.

Per mode:
  enroll   --encoder PATH  --seconds N  --wav PATH  --out-wav PATH
  bench    --iters N
  file     -i PATH  -o PATH
  verify   --emb-target PATH  --emb-other PATH  -i PATH
  live     --save PREFIX  --in-device NAME  --out-device NAME

Configuration:
  Put the values you use often into voice_lock.conf and stop typing them.
  Precedence: defaults < voice_lock.conf < command line.

Examples:
  tse_voice_lock live                    # everything from voice_lock.conf
  tse_voice_lock live --power 4          # same, but override power
  tse_voice_lock bench --no-config       # ignore the config file
)");
}

Args Parse(int argc, char** argv) {
  if (argc < 2) {
    Usage();
    std::exit(1);
  }

  Args a;
  a.cmd = argv[1];

  auto need = [&](int i) -> const char* {
    if (i + 1 >= argc) Die(std::string("Missing value for ") + argv[i]);
    return argv[i + 1];
  };

  // --- Pass 1: look only for --config / --no-config ------------------------
  std::string config_path = "voice_lock.conf";
  bool use_config = true;

  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--config") {
      config_path = need(i);
      ++i;
    } else if (k == "--no-config") {
      use_config = false;
    }
  }

  if (use_config) {
    const std::string loaded = LoadConfig(config_path, a);

    if (!loaded.empty()) {
      a.config_used = loaded;
    } else if (config_path != "voice_lock.conf") {
      // An explicit --config that cannot be found must be reported, not
      // silently ignored.
      Die("Config file not found: " + config_path);
    }
  }

  // --- Pass 2: command line arguments, overriding the config --------------
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];

    if      (k == "--config")     ++i;            // handled in pass 1
    else if (k == "--no-config")  ;               // handled in pass 1
    else if (k == "--model")      a.opt.model   = need(i), ++i;
    else if (k == "--emb")        a.opt.emb     = need(i), ++i;
    else if (k == "--power")      a.opt.power   = std::atof(need(i)), ++i;
    else if (k == "--gain")       a.opt.gain_db = std::atof(need(i)), ++i;
    else if (k == "--lock-db")    a.opt.lock_db = std::atof(need(i)), ++i;
    else if (k == "--threads")    a.opt.threads = std::atoi(need(i)), ++i;
    else if (k == "--norm") {
      const std::string v = need(i);
      ++i;
      if (v != "off" && v != "peak") Die("--norm accepts only 'off' or 'peak'");
      a.opt.norm_peak = v == "peak";
    }
    else if (k == "--encoder")    a.encoder     = need(i), ++i;
    else if (k == "--save")       a.save        = need(i), ++i;
    else if (k == "--seconds")    a.seconds     = std::atof(need(i)), ++i;
    else if (k == "--iters")      a.iters       = std::atoi(need(i)), ++i;
    else if (k == "--wav")        a.input       = need(i), ++i;
    else if (k == "--out-wav")    a.out_wav     = need(i), ++i;
    else if (k == "--emb-target") a.emb_target  = need(i), ++i;
    else if (k == "--emb-other")  a.emb_other   = need(i), ++i;
    else if (k == "--in-device")  a.in_device   = need(i), ++i;
    else if (k == "--out-device") a.out_device  = need(i), ++i;
    else if (k == "-i" || k == "--input")  a.input  = need(i), ++i;
    else if (k == "-o" || k == "--output") a.output = need(i), ++i;
    else if (k == "-h" || k == "--help") { Usage(); std::exit(0); }
    else Die("Unknown option: " + k);
  }

  if (a.opt.threads < 1) a.opt.threads = 1;

  return a;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

VoiceLock Build(const Args& a) {
  std::printf("==============================================================\n");
  std::printf("V49 VOICE LOCK (C++)\n");
  std::printf("==============================================================\n");
  std::printf("  model : %s\n", a.opt.model.c_str());
  std::printf("  emb   : %s\n", a.opt.emb.c_str());
  std::printf("  power : %.1f   gain : +%.1f dB   lock : %+.1f dB\n",
              a.opt.power, a.opt.gain_db, a.opt.lock_db);
  if (!a.config_used.empty()) {
    std::printf("  config: %s\n", a.config_used.c_str());
  }

  return VoiceLock(a.opt, LoadEmbedding(a.opt.emb));
}

std::vector<float> LoadAudio16k(const std::string& path) {
  int sr = 0;
  std::vector<float> a = ReadWav(path, &sr);

  if (sr != kSampleRate) {
    Die("File " + path + " is " + std::to_string(sr) +
        " Hz, the model needs 16000 Hz.\n"
        "  Convert it: sox in.wav -r 16000 -c 1 out.wav");
  }

  return a;
}

void Summary(const VoiceLock& vl, const std::vector<ChunkStats>& all) {
  std::vector<const ChunkStats*> v;
  for (const auto& s : all) {
    if (s.state != LockState::kSilence) v.push_back(&s);
  }

  std::printf("\n==============================================================\n");
  std::printf("SUMMARY\n");
  std::printf("==============================================================\n");

  if (v.empty()) {
    std::printf("  No chunk contained any sound.\n");
    return;
  }

  double sum = 0.0, mask_sum = 0.0;
  float lo = 1e9f, hi = -1e9f;
  int locked = 0;

  for (const auto* s : v) {
    sum += s->suppress_db;
    mask_sum += s->mask;
    lo = std::min(lo, s->suppress_db);
    hi = std::max(hi, s->suppress_db);
    if (s->state == LockState::kLocked) ++locked;
  }

  const double n = static_cast<double>(v.size());
  const float mask_avg = static_cast<float>(mask_sum / n);

  std::printf("  chunks with sound : %zu\n", v.size());
  std::printf("  LOCKED            : %d (%.0f%%)\n", locked, 100.0 * locked / n);
  std::printf("  REJECT            : %zu\n", v.size() - locked);
  std::printf("  suppression       : mean %+.1f dB, min %+.1f / max %+.1f\n",
              sum / n, lo, hi);
  std::printf("  model mask        : mean %.3f  (%s)\n",
              mask_avg, MaskVerdict(mask_avg));
  std::printf("  latency           : %.0f ms/chunk, RTF %.3f (%s)\n",
              vl.total_ms() / std::max(vl.processed(), 1), vl.rtf(),
              vl.rtf() < 1.0 ? "realtime OK" : "TOO SLOW");
}

// ---------------------------------------------------------------------------
// selftest - no model needed, used to cross-check against the Python port
// ---------------------------------------------------------------------------

int CmdSelftest() {
  std::printf("SELFTEST - DSP only, no ONNX model required\n\n");

  // Deterministic signal: sin 440 Hz + sin 1234 Hz. The Python side must
  // generate exactly the same thing (see tools/parity_ref.py).
  constexpr double kPi = 3.14159265358979323846;

  const int n = 48000;
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / kSampleRate;
    x[i] = static_cast<float>(0.5 * std::sin(2.0 * kPi * 440.0 * t) +
                              0.25 * std::sin(2.0 * kPi * 1234.0 * t));
  }

  Stft stft;
  const Spectrogram s = stft.Forward(x.data(), n);

  std::printf("  frames        : %d  (expected 376)\n", s.frames);

  std::vector<float> y(n);
  stft.Inverse(s.real.data(), s.imag.data(), s.frames, y.data(), n);

  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) max_err = std::max(max_err, std::fabs(x[i] - y[i]));

  double mag_sum = 0.0;
  for (float v : s.mag) mag_sum += v;

  std::vector<float> norm;
  NormalizeInput(s.mag, norm);

  double norm_mean = 0.0;
  for (float v : norm) norm_mean += v;
  norm_mean /= norm.size();

  // Print enough significant digits for tools/parity_ref.py to really
  // compare.
  //
  // Bin 14 = 440 Hz (440 / (16000/512) = 14.08) - pick a bin that HAS
  // energy. An empty bin (say bin 100) is just float32 noise floor, where
  // two different FFT algorithms differ by ~1e-4 relative for no meaningful
  // reason.
  std::printf("  roundtrip err : %.6e   (must be < 1e-5)\n", max_err);
  std::printf("  sum(mag)      : %.9g\n", mag_sum);
  std::printf("  mag[14,50]    : %.9g\n",
              s.mag[static_cast<size_t>(14) * s.frames + 50]);
  std::printf("  norm mean     : %.6e\n", norm_mean);

  std::vector<float> mv;
  NormalizeMeanVar(x.data(), n, mv);
  std::printf("  meanvar[1000] : %.9g\n", mv[1000]);

  const bool ok = max_err < 1e-5f && s.frames == 376;

  std::printf("\n  %s\n",
              ok ? "PASS" : "FAIL - DSP is off, do not run the model");
  std::printf("\n  Cross-check with: python tools/parity_ref.py\n");

  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// enroll
// ---------------------------------------------------------------------------

int CmdEnroll(const Args& a) {
  std::printf("==============================================================\n");
  std::printf("ENROLLMENT - build a 512-d speaker embedding\n");
  std::printf("==============================================================\n");

  // --encoder is optional: search models/ the same way the Python port does.
  std::string encoder = ResolvePath(a.encoder.empty() ? "wavlm_sv_int8.onnx"
                                                      : a.encoder);
  if (encoder.empty() && a.encoder.empty()) {
    encoder = ResolvePath("wavlm_sv_fp32.onnx");
  }

  if (encoder.empty()) {
    Die("No WavLM ONNX encoder found.\n"
        "  Searched: models/wavlm_sv_int8.onnx , models/wavlm_sv_fp32.onnx\n"
        "  Build it on a laptop: python export_wavlm_onnx.py --outdir models\n"
        "  Or enroll on the laptop and copy speaker_emb.npy over.");
  }

  std::vector<float> audio;

  if (!a.input.empty()) {
    audio = LoadAudio16k(a.input);
    std::printf("  read: %s (%.1f s)\n", a.input.c_str(),
                audio.size() / static_cast<double>(kSampleRate));
  } else {
    const int total = static_cast<int>(a.seconds * kSampleRate);

    std::printf("\n  Recording %.0f s. Speak naturally, vary your intonation.\n",
                a.seconds);
    for (int i = 3; i > 0; --i) {
      std::printf("  %d...\n", i);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::printf("  GO\n\n");

    AlsaDevice mic(a.in_device, kSampleRate, 4096, true);

    audio.resize(total);
    if (!mic.Read(audio.data(), total)) Die("Reading from the mic failed");

    WriteWav(a.out_wav, audio, kSampleRate);
    std::printf("  saved: %s\n", a.out_wav.c_str());
  }

  const float r = Rms(audio.data(), static_cast<int>(audio.size()));
  float peak = 0.0f;
  for (float v : audio) peak = std::max(peak, std::fabs(v));

  std::printf("  RMS=%.4f  peak=%.4f\n", r, peak);
  if (r < 0.005f) {
    std::printf("  WARNING: very quiet - check the mic (alsamixer)\n");
  }
  if (peak > 0.95f) {
    std::printf("  WARNING: clipping - move further from the mic\n");
  }

  const EnrollResult res = ComputeEmbedding(encoder, audio, a.opt.threads);

  WriteNpyFloat32(a.opt.emb, res.embedding);

  std::printf("\n  embedding: (512,) -> %s\n", a.opt.emb.c_str());
  std::printf("  %d segments, consistency %.3f (>0.85 good, <0.7 noisy)\n",
              res.segments, res.consistency);

  if (res.consistency < 0.7f) {
    std::printf("  Consider re-recording somewhere quieter.\n");
  }

  return 0;
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

int CmdBench(const Args& a) {
  VoiceLock vl = Build(a);

  std::mt19937 rng(12345);
  std::normal_distribution<float> dist(0.0f, 0.05f);

  std::vector<float> dummy(vl.chunk_samples());
  for (float& v : dummy) v = dist(rng);

  std::vector<float> out;

  std::printf("\n  Warmup...\n");
  for (int i = 0; i < 3; ++i) vl.Process(dummy.data(), dummy.size(), &out);

  vl.ResetStats();

  std::vector<double> ms;
  ms.reserve(a.iters);

  for (int i = 0; i < a.iters; ++i) {
    ms.push_back(vl.Process(dummy.data(), dummy.size(), &out).ms);
  }

  std::sort(ms.begin(), ms.end());

  double sum = 0.0;
  for (double v : ms) sum += v;
  const double mean = sum / ms.size();

  std::printf("\n  Benchmark (%d runs, %.1f s chunks, %d threads)\n",
              a.iters, vl.chunk_sec(), a.opt.threads);
  std::printf("    mean   : %7.0f ms\n", mean);
  std::printf("    median : %7.0f ms\n", ms[ms.size() / 2]);
  std::printf("    min/max: %.0f / %.0f ms\n", ms.front(), ms.back());
  std::printf("    RTF    : %.3f   %s\n", vl.rtf(),
              vl.rtf() < 1.0 ? "REALTIME OK" : "TOO SLOW");

  const double budget = vl.hop_samples() * 1000.0 / kSampleRate;
  std::printf("\n    Budget per hop = %.0f ms -> %s\n", budget,
              mean < budget ? "enough" : "NOT ENOUGH, audio will drop");

  if (vl.rtf() >= 1.0) {
    std::printf("\n  Ways to speed this up on a Pi 5:\n");
    std::printf("    - --power 1 (drops sharpening, ~20%% faster)\n");
    std::printf("    - sudo cpufreq-set -g performance\n");
  }

  return 0;
}

// ---------------------------------------------------------------------------
// file
// ---------------------------------------------------------------------------

int CmdFile(const Args& a) {
  if (a.input.empty()) Die("-i <file.wav> is required");

  VoiceLock vl = Build(a);

  std::vector<float> audio = LoadAudio16k(a.input);

  const int win = vl.chunk_samples();
  const int hop = vl.hop_samples();

  if (static_cast<int>(audio.size()) < win) audio.resize(win, 0.0f);

  const int total = static_cast<int>(audio.size());

  std::vector<float> out(total, 0.0f), weight(total, 0.0f);
  const std::vector<float>& fade = vl.fade_window();

  std::printf("\n  %s (%.1f s)\n", a.input.c_str(),
              total / static_cast<double>(kSampleRate));
  std::printf("\n     t (s)    in dB    out dB   suppress   mask      ms  state\n");
  std::printf("  ------------------------------------------------------------------\n");

  std::vector<ChunkStats> all;
  std::vector<float> chunk_out;

  for (int start = 0; start + win <= total; start += hop) {
    const ChunkStats st = vl.Process(audio.data() + start, win, &chunk_out);
    all.push_back(st);

    for (int i = 0; i < win; ++i) {
      out[start + i]    += chunk_out[i] * fade[i];
      weight[start + i] += fade[i];
    }

    std::printf("  %8.1f %8.1f %9.1f %+9.1f dB %6.3f %6.0f  %s\n",
                start / static_cast<double>(kSampleRate), st.in_db, st.out_db,
                st.suppress_db, st.mask, st.ms, ToString(st.state));
  }

  float peak = 0.0f;
  for (int i = 0; i < total; ++i) {
    out[i] = weight[i] > 1e-8f ? out[i] / weight[i] : 0.0f;
    peak = std::max(peak, std::fabs(out[i]));
  }

  if (a.opt.norm_peak) {
    // One gain for the whole file, applied after suppression was measured,
    // so the figures stay honest and LOCKED still sits above REJECT.
    NormalizePeak(&out);
    std::printf("\n  (--norm peak: %.4f -> 0.9500)\n", peak);
  } else if (peak > 1.0f) {
    for (float& v : out) v /= peak;
    std::printf("\n  (rescaled to avoid clipping, peak was %.2f)\n", peak);
  }

  WriteWav(a.output, out, kSampleRate);

  Summary(vl, all);
  std::printf("\n  Saved: %s\n", a.output.c_str());

  return 0;
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

int CmdVerify(const Args& a) {
  if (a.input.empty()) Die("-i <file.wav> is required");
  if (a.emb_target.empty() || a.emb_other.empty()) {
    Die("--emb-target and --emb-other are required");
  }

  const std::vector<float> audio = LoadAudio16k(a.input);

  double result[2] = {0.0, 0.0};
  const char* label[2] = {"TARGET", "OTHER "};
  const std::string path[2] = {a.emb_target, a.emb_other};

  for (int k = 0; k < 2; ++k) {
    std::printf("\n  --- embedding: %s (%s) ---\n", label[k], path[k].c_str());

    Options o = a.opt;
    o.emb = path[k];

    VoiceLock vl(o, LoadEmbedding(path[k]));

    const int win = vl.chunk_samples();
    const int hop = vl.hop_samples();

    std::vector<float> pad = audio;
    if (static_cast<int>(pad.size()) < win) pad.resize(win, 0.0f);

    std::vector<float> out;
    double sum = 0.0;
    int n = 0;

    for (int s = 0; s + win <= static_cast<int>(pad.size()); s += hop) {
      const ChunkStats st = vl.Process(pad.data() + s, win, &out);
      if (st.state != LockState::kSilence) {
        sum += st.suppress_db;
        ++n;
      }
    }

    result[k] = n ? sum / n : 0.0;
    std::printf("      mean suppression: %+.1f dB\n", result[k]);
  }

  const double gap = result[0] - result[1];

  std::printf("\n==============================================================\n");
  std::printf("VOICE LOCK RESULT\n");
  std::printf("==============================================================\n");
  std::printf("  correct voice (target) : %+6.1f dB  (closer to 0 is better)\n",
              result[0]);
  std::printf("  wrong voice   (other)  : %+6.1f dB  (more negative is better)\n",
              result[1]);
  std::printf("  gap                    : %+6.1f dB\n", gap);

  if (gap > 6.0) {
    std::printf("\n  => GOOD LOCK: the model clearly follows the enrollment.\n");
  } else if (gap > 3.0) {
    std::printf("\n  => WEAK LOCK: re-enroll for 10-15 s, raise --power.\n");
  } else {
    std::printf("\n  => NO LOCK: check the embedding and the input file.\n");
  }

  return 0;
}

// ---------------------------------------------------------------------------
// live
// ---------------------------------------------------------------------------

int CmdLive(const Args& a) {
  VoiceLock vl = Build(a);

  const int chunk = vl.chunk_samples();
  const int hop   = vl.hop_samples();
  const std::vector<float>& fade = vl.fade_window();

  AlsaDevice mic(a.in_device, kSampleRate, hop, true);
  AlsaDevice spk(a.out_device, kSampleRate, hop, false);

  std::printf("\n  chunk %.1f s / hop %.1f s, lock threshold %+.1f dB\n",
              vl.chunk_sec(), hop / static_cast<double>(kSampleRate),
              a.opt.lock_db);
  std::printf("  WEAR HEADPHONES - speakers will cause feedback.\n");
  if (!a.save.empty()) {
    std::printf("  recording to: %s_input.wav / %s_output.wav\n",
                a.save.c_str(), a.save.c_str());
  }
  std::printf("  Ctrl+C to stop.\n\n");
  std::printf("   chunk    in dB    out dB   suppress   mask      ms    RTF  state\n");
  std::printf("  ----------------------------------------------------------------------\n");

  std::vector<float> in_buf(chunk, 0.0f);
  std::vector<float> out_buf(chunk, 0.0f);
  std::vector<float> out_w(chunk, 0.0f);
  std::vector<float> block(hop), emit(hop), extracted;

  std::vector<float> rec_in, rec_out;
  std::vector<ChunkStats> all;

  std::signal(SIGINT, OnSigint);

  // Prime the playback buffer so the first chunk does not underrun
  spk.Prefill(hop);

  while (!g_stop) {
    if (!mic.Read(block.data(), hop)) {
      std::fprintf(stderr, "\n  Lost the microphone.\n");
      break;
    }

    if (!a.save.empty()) rec_in.insert(rec_in.end(), block.begin(), block.end());

    // Slide the input window: drop the oldest hop, append the new one
    std::memmove(in_buf.data(), in_buf.data() + hop,
                 sizeof(float) * (chunk - hop));
    std::memcpy(in_buf.data() + chunk - hop, block.data(), sizeof(float) * hop);

    const ChunkStats st = vl.Process(in_buf.data(), chunk, &extracted);
    all.push_back(st);

    for (int i = 0; i < chunk; ++i) {
      out_buf[i] += extracted[i] * fade[i];
      out_w[i]   += fade[i];
    }

    for (int i = 0; i < hop; ++i) {
      float v = out_w[i] > 1e-8f ? out_buf[i] / out_w[i] : 0.0f;
      emit[i] = std::max(-1.0f, std::min(1.0f, v));
    }

    std::memmove(out_buf.data(), out_buf.data() + hop,
                 sizeof(float) * (chunk - hop));
    std::memset(out_buf.data() + chunk - hop, 0, sizeof(float) * hop);
    std::memmove(out_w.data(), out_w.data() + hop,
                 sizeof(float) * (chunk - hop));
    std::memset(out_w.data() + chunk - hop, 0, sizeof(float) * hop);

    if (!spk.Write(emit.data(), hop)) {
      std::fprintf(stderr, "\n  Lost the speaker.\n");
      break;
    }

    if (!a.save.empty()) rec_out.insert(rec_out.end(), emit.begin(), emit.end());

    std::printf("  %6d %8.1f %9.1f %+9.1f dB %6.3f %6.0f %6.3f  %s\n",
                vl.processed(), st.in_db, st.out_db, st.suppress_db, st.mask,
                st.ms, vl.rtf(), ToString(st.state));
    std::fflush(stdout);
  }

  std::printf("\n  Stopped. xruns: mic %d, speaker %d\n",
              mic.xruns(), spk.xruns());

  Summary(vl, all);

  if (!a.save.empty() && !rec_in.empty()) {
    const size_t n = std::min(rec_in.size(), rec_out.size());
    rec_in.resize(n);
    rec_out.resize(n);

    if (a.opt.norm_peak) {
      // Only the saved file is normalized - this does not change what was
      // played live, nor the suppression figures.
      NormalizePeak(&rec_out);
      std::printf("  (--norm peak applied to the output file)\n");
    }

    WriteWav(a.save + "_input.wav", rec_in, kSampleRate);
    WriteWav(a.save + "_output.wav", rec_out, kSampleRate);

    std::printf("\n  Saved %.1f s:\n", n / static_cast<double>(kSampleRate));
    std::printf("    %s_input.wav\n", a.save.c_str());
    std::printf("    %s_output.wav\n", a.save.c_str());
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);

  try {
    if (a.cmd == "selftest") return CmdSelftest();
    if (a.cmd == "devices")  { ListDevices(); return 0; }
    if (a.cmd == "enroll")   return CmdEnroll(a);
    if (a.cmd == "bench")    return CmdBench(a);
    if (a.cmd == "file")     return CmdFile(a);
    if (a.cmd == "verify")   return CmdVerify(a);
    if (a.cmd == "live")     return CmdLive(a);

    Usage();
    return 1;

  } catch (const Ort::Exception& e) {
    std::fprintf(stderr, "\nONNX Runtime error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nERROR: %s\n", e.what());
    return 1;
  }
}
