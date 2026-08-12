// main.cpp — CLI cho V49 voice lock, ban C++.
//
// Cac mode giu nguyen ten va y nghia nhu ban Python (voice_lock.py) de
// hai ban co the doi chieu truc tiep voi nhau.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Phan tich tham so
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
};

[[noreturn]] void Die(const std::string& msg) {
  std::fprintf(stderr, "\nLOI: %s\n", msg.c_str());
  std::exit(1);
}

void Usage() {
  std::printf(R"(V49 VOICE LOCK — ban C++ cho Raspberry Pi 5

  tse_voice_lock <mode> [tham so]

Mode:
  enroll    tao speaker embedding tu mic hoac file wav
  bench     do latency / RTF
  file      tach giong tu file wav
  verify    A/B giong dung vs giong sai
  live      realtime qua mic (deo tai nghe)
  devices   liet ke thiet bi ALSA
  selftest  kiem tra STFT/iSTFT, khong can model

Tham so chung:
  --model PATH      mac dinh v49_int8.onnx (tu tim trong models/)
  --emb PATH        mac dinh speaker_emb.npy
  --power N         mask sharpening (1=tat, 2=mac dinh, 3=manh)
  --gain N          gain dau ra dB
  --lock-db N       nguong LOCKED (mac dinh -8)
  --threads N       so thread ONNX (mac dinh 4)

Rieng tung mode:
  enroll   --encoder PATH  --seconds N  --wav PATH  --out-wav PATH
  bench    --iters N
  file     -i PATH  -o PATH
  verify   --emb-target PATH  --emb-other PATH  -i PATH
  live     --save PREFIX  --in-device NAME  --out-device NAME

Vi du:
  tse_voice_lock enroll --seconds 10
  tse_voice_lock bench
  tse_voice_lock live   --emb speaker_emb.npy --power 3 --save demo1
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
    if (i + 1 >= argc) Die(std::string("Thieu gia tri cho ") + argv[i]);
    return argv[i + 1];
  };

  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];

    if      (k == "--model")      a.opt.model   = need(i), ++i;
    else if (k == "--emb")        a.opt.emb     = need(i), ++i;
    else if (k == "--power")      a.opt.power   = std::atof(need(i)), ++i;
    else if (k == "--gain")       a.opt.gain_db = std::atof(need(i)), ++i;
    else if (k == "--lock-db")    a.opt.lock_db = std::atof(need(i)), ++i;
    else if (k == "--threads")    a.opt.threads = std::atoi(need(i)), ++i;
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
    else Die("Tham so la: " + k);
  }

  if (a.opt.threads < 1) a.opt.threads = 1;

  return a;
}

// ---------------------------------------------------------------------------
// Tien ich chung
// ---------------------------------------------------------------------------

VoiceLock Build(const Args& a) {
  std::printf("==============================================================\n");
  std::printf("V49 VOICE LOCK (C++)\n");
  std::printf("==============================================================\n");
  std::printf("  model : %s\n", a.opt.model.c_str());
  std::printf("  emb   : %s\n", a.opt.emb.c_str());
  std::printf("  power : %.1f   gain : +%.1f dB   lock : %+.1f dB\n",
              a.opt.power, a.opt.gain_db, a.opt.lock_db);

  return VoiceLock(a.opt, LoadEmbedding(a.opt.emb));
}

std::vector<float> LoadAudio16k(const std::string& path) {
  int sr = 0;
  std::vector<float> a = ReadWav(path, &sr);

  if (sr != kSampleRate) {
    Die("File " + path + " la " + std::to_string(sr) + " Hz, model can 16000 Hz.\n"
        "  Convert: sox in.wav -r 16000 -c 1 out.wav");
  }

  return a;
}

void Summary(const VoiceLock& vl, const std::vector<ChunkStats>& all) {
  std::vector<const ChunkStats*> v;
  for (const auto& s : all) {
    if (s.state != LockState::kSilence) v.push_back(&s);
  }

  std::printf("\n==============================================================\n");
  std::printf("TONG KET\n");
  std::printf("==============================================================\n");

  if (v.empty()) {
    std::printf("  Khong co chunk nao co tieng.\n");
    return;
  }

  double sum = 0.0;
  float lo = 1e9f, hi = -1e9f;
  int locked = 0;

  for (const auto* s : v) {
    sum += s->suppress_db;
    lo = std::min(lo, s->suppress_db);
    hi = std::max(hi, s->suppress_db);
    if (s->state == LockState::kLocked) ++locked;
  }

  const double n = static_cast<double>(v.size());

  std::printf("  chunk co tieng : %zu\n", v.size());
  std::printf("  LOCKED         : %d (%.0f%%)\n", locked, 100.0 * locked / n);
  std::printf("  REJECT         : %zu\n", v.size() - locked);
  std::printf("  suppress       : trung binh %+.1f dB, min %+.1f / max %+.1f\n",
              sum / n, lo, hi);
  std::printf("  latency        : %.0f ms/chunk, RTF %.3f (%s)\n",
              vl.total_ms() / std::max(vl.processed(), 1), vl.rtf(),
              vl.rtf() < 1.0 ? "realtime OK" : "QUA CHAM");
}

// ---------------------------------------------------------------------------
// selftest — khong can model, dung de doi chieu voi ban Python
// ---------------------------------------------------------------------------

int CmdSelftest() {
  std::printf("SELFTEST — DSP, khong can model ONNX\n\n");

  // Tin hieu tat dinh: sin 440 Hz + sin 1234 Hz. Ban Python phai
  // sinh y het (xem tools/parity_ref.py).
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

  std::printf("  frames        : %d  (mong doi 376)\n", s.frames);

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

  // In du chu so y nghia de tools/parity_ref.py so sanh duoc that su.
  //
  // Bin 14 = 440 Hz (440 / (16000/512) = 14.08) — chon bin CO nang luong.
  // Bin trong (vd bin 100) chi la san nhieu float32, hai thuat toan FFT
  // khac nhau se lech ~1e-4 tuong doi o do ma khong co y nghia gi.
  std::printf("  roundtrip err : %.6e   (can < 1e-5)\n", max_err);
  std::printf("  sum(mag)      : %.9g\n", mag_sum);
  std::printf("  mag[14,50]    : %.9g\n",
              s.mag[static_cast<size_t>(14) * s.frames + 50]);
  std::printf("  norm mean     : %.6e\n", norm_mean);

  std::vector<float> mv;
  NormalizeMeanVar(x.data(), n, mv);
  std::printf("  meanvar[1000] : %.9g\n", mv[1000]);

  const bool ok = max_err < 1e-5f && s.frames == 376;

  std::printf("\n  %s\n", ok ? "PASS" : "FAIL — DSP lech, khong chay model");
  std::printf("\n  Doi chieu: python tools/parity_ref.py\n");

  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// enroll
// ---------------------------------------------------------------------------

int CmdEnroll(const Args& a) {
  std::printf("==============================================================\n");
  std::printf("ENROLLMENT — tao speaker embedding 512-d\n");
  std::printf("==============================================================\n");

  // Khong ep phai go --encoder: tu tim trong models/ nhu ban Python.
  std::string encoder = ResolvePath(a.encoder.empty() ? "wavlm_sv_int8.onnx"
                                                      : a.encoder);
  if (encoder.empty() && a.encoder.empty()) {
    encoder = ResolvePath("wavlm_sv_fp32.onnx");
  }

  if (encoder.empty()) {
    Die("Khong tim thay WavLM encoder ONNX.\n"
        "  Da tim: models/wavlm_sv_int8.onnx , models/wavlm_sv_fp32.onnx\n"
        "  Tao tren laptop: python export_wavlm_onnx.py --outdir models\n"
        "  Hoac enroll tren laptop roi copy speaker_emb.npy sang day.");
  }

  std::vector<float> audio;

  if (!a.input.empty()) {
    audio = LoadAudio16k(a.input);
    std::printf("  doc: %s (%.1f s)\n", a.input.c_str(),
                audio.size() / static_cast<double>(kSampleRate));
  } else {
    const int total = static_cast<int>(a.seconds * kSampleRate);

    std::printf("\n  Ghi am %.0f s. Noi tu nhien, thay doi ngu dieu.\n",
                a.seconds);
    for (int i = 3; i > 0; --i) {
      std::printf("  %d...\n", i);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::printf("  GO\n\n");

    AlsaDevice mic(a.in_device, kSampleRate, 4096, true);

    audio.resize(total);
    if (!mic.Read(audio.data(), total)) Die("Doc mic that bai");

    WriteWav(a.out_wav, audio, kSampleRate);
    std::printf("  luu: %s\n", a.out_wav.c_str());
  }

  const float r = Rms(audio.data(), static_cast<int>(audio.size()));
  float peak = 0.0f;
  for (float v : audio) peak = std::max(peak, std::fabs(v));

  std::printf("  RMS=%.4f  peak=%.4f\n", r, peak);
  if (r < 0.005f) std::printf("  CANH BAO: qua nho — kiem tra mic (alsamixer)\n");
  if (peak > 0.95f) std::printf("  CANH BAO: clipping — lui ra xa mic\n");

  const EnrollResult res = ComputeEmbedding(encoder, audio, a.opt.threads);

  WriteNpyFloat32(a.opt.emb, res.embedding);

  std::printf("\n  embedding: (512,) -> %s\n", a.opt.emb.c_str());
  std::printf("  %d segment, consistency %.3f (>0.85 tot, <0.7 nhieu)\n",
              res.segments, res.consistency);

  if (res.consistency < 0.7f) {
    std::printf("  Nen ghi lai o noi yen tinh hon.\n");
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

  std::printf("\n  Benchmark (%d lan, chunk %.1f s, %d threads)\n",
              a.iters, vl.chunk_sec(), a.opt.threads);
  std::printf("    mean   : %7.0f ms\n", mean);
  std::printf("    median : %7.0f ms\n", ms[ms.size() / 2]);
  std::printf("    min/max: %.0f / %.0f ms\n", ms.front(), ms.back());
  std::printf("    RTF    : %.3f   %s\n", vl.rtf(),
              vl.rtf() < 1.0 ? "REALTIME OK" : "QUA CHAM");

  const double budget = vl.hop_samples() * 1000.0 / kSampleRate;
  std::printf("\n    Ngan sach 1 hop = %.0f ms -> %s\n", budget,
              mean < budget ? "du" : "THIEU, se drop audio");

  if (vl.rtf() >= 1.0) {
    std::printf("\n  Cach tang toc tren Pi 5:\n");
    std::printf("    - --power 1 (bo sharpening, ~20%% nhanh hon)\n");
    std::printf("    - sudo cpufreq-set -g performance\n");
  }

  return 0;
}

// ---------------------------------------------------------------------------
// file
// ---------------------------------------------------------------------------

int CmdFile(const Args& a) {
  if (a.input.empty()) Die("Can -i <file.wav>");

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
  std::printf("\n     t (s)    in dB    out dB   suppress      ms  state\n");
  std::printf("  ----------------------------------------------------------\n");

  std::vector<ChunkStats> all;
  std::vector<float> chunk_out;

  for (int start = 0; start + win <= total; start += hop) {
    const ChunkStats st = vl.Process(audio.data() + start, win, &chunk_out);
    all.push_back(st);

    for (int i = 0; i < win; ++i) {
      out[start + i]    += chunk_out[i] * fade[i];
      weight[start + i] += fade[i];
    }

    std::printf("  %8.1f %8.1f %9.1f %+9.1f dB %6.0f  %s\n",
                start / static_cast<double>(kSampleRate), st.in_db, st.out_db,
                st.suppress_db, st.ms, ToString(st.state));
  }

  float peak = 0.0f;
  for (int i = 0; i < total; ++i) {
    out[i] = weight[i] > 1e-8f ? out[i] / weight[i] : 0.0f;
    peak = std::max(peak, std::fabs(out[i]));
  }

  if (peak > 1.0f) {
    for (float& v : out) v /= peak;
    std::printf("\n  (chuan hoa lai, peak was %.2f)\n", peak);
  }

  WriteWav(a.output, out, kSampleRate);

  Summary(vl, all);
  std::printf("\n  Da luu: %s\n", a.output.c_str());

  return 0;
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

int CmdVerify(const Args& a) {
  if (a.input.empty()) Die("Can -i <file.wav>");
  if (a.emb_target.empty() || a.emb_other.empty()) {
    Die("Can --emb-target va --emb-other");
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
    std::printf("      suppression trung binh: %+.1f dB\n", result[k]);
  }

  const double gap = result[0] - result[1];

  std::printf("\n==============================================================\n");
  std::printf("KET QUA VOICE LOCK\n");
  std::printf("==============================================================\n");
  std::printf("  giong dung (target) : %+6.1f dB  (cang gan 0 cang tot)\n", result[0]);
  std::printf("  giong sai  (other)  : %+6.1f dB  (cang am cang tot)\n", result[1]);
  std::printf("  khoang cach         : %+6.1f dB\n", gap);

  if (gap > 6.0) {
    std::printf("\n  => LOCK TOT: model phan biet ro theo enrollment.\n");
  } else if (gap > 3.0) {
    std::printf("\n  => LOCK YEU: nen enroll lai 10-15 s, tang --power.\n");
  } else {
    std::printf("\n  => KHONG LOCK: kiem tra lai embedding / file dau vao.\n");
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

  std::printf("\n  chunk %.1f s / hop %.1f s, nguong lock %+.1f dB\n",
              vl.chunk_sec(), hop / static_cast<double>(kSampleRate),
              a.opt.lock_db);
  std::printf("  DEO TAI NGHE — dung loa se bi hu (feedback).\n");
  if (!a.save.empty()) {
    std::printf("  ghi ra: %s_input.wav / %s_output.wav\n",
                a.save.c_str(), a.save.c_str());
  }
  std::printf("  Ctrl+C de dung.\n\n");
  std::printf("   chunk    in dB    out dB   suppress      ms    RTF  state\n");
  std::printf("  --------------------------------------------------------------\n");

  std::vector<float> in_buf(chunk, 0.0f);
  std::vector<float> out_buf(chunk, 0.0f);
  std::vector<float> out_w(chunk, 0.0f);
  std::vector<float> block(hop), emit(hop), extracted;

  std::vector<float> rec_in, rec_out;
  std::vector<ChunkStats> all;

  std::signal(SIGINT, OnSigint);

  // Mo san buffer phat de khoi underrun ngay chunk dau
  spk.Prefill(hop);

  while (!g_stop) {
    if (!mic.Read(block.data(), hop)) {
      std::fprintf(stderr, "\n  Mic dut ket noi.\n");
      break;
    }

    if (!a.save.empty()) rec_in.insert(rec_in.end(), block.begin(), block.end());

    // Truot cua so vao: bo hop cu nhat, nap hop moi
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
      std::fprintf(stderr, "\n  Loa dut ket noi.\n");
      break;
    }

    if (!a.save.empty()) rec_out.insert(rec_out.end(), emit.begin(), emit.end());

    std::printf("  %6d %8.1f %9.1f %+9.1f dB %6.0f %6.3f  %s\n",
                vl.processed(), st.in_db, st.out_db, st.suppress_db, st.ms,
                vl.rtf(), ToString(st.state));
    std::fflush(stdout);
  }

  std::printf("\n  Dung. xrun: mic %d, loa %d\n", mic.xruns(), spk.xruns());

  Summary(vl, all);

  if (!a.save.empty() && !rec_in.empty()) {
    const size_t n = std::min(rec_in.size(), rec_out.size());
    rec_in.resize(n);
    rec_out.resize(n);

    WriteWav(a.save + "_input.wav", rec_in, kSampleRate);
    WriteWav(a.save + "_output.wav", rec_out, kSampleRate);

    std::printf("\n  Da luu %.1f s:\n", n / static_cast<double>(kSampleRate));
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
    std::fprintf(stderr, "\nLOI ONNX Runtime: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nLOI: %s\n", e.what());
    return 1;
  }
}
