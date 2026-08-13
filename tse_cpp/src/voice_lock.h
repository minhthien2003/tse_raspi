// voice_lock.h - runs the V49 ONNX model and measures how strongly the
// incoming voice is locked to the enrolled speaker.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "core.h"

namespace tse {

enum class LockState { kSilence, kLocked, kReject };

const char* ToString(LockState s);

struct ChunkStats {
  LockState state       = LockState::kSilence;
  float     in_db       = -99.0f;
  float     out_db      = -99.0f;
  float     suppress_db = 0.0f;
  float     mask        = 0.0f;   // mask the model produced, BEFORE power
  double    ms          = 0.0;
};

// Interprets the mean mask: separates "needs tuning" from "bad embedding".
const char* MaskVerdict(float mask);

// Normalizes output loudness. Applies ONE gain to the whole signal - a
// per-chunk normalizer would lift REJECT segments up to LOCKED level and
// destroy the very effect voice lock provides. Always call AFTER suppression
// has been measured.
void NormalizePeak(std::vector<float>* audio, float target_peak = 0.95f);

// Looks for a file in the usual places: verbatim first, then models/ in the
// current directory, its parent, and its grandparent. This lets the binary
// run from tse_cpp/build/ and still find models/ at the project root.
// Returns an empty string when nothing matches.
std::string ResolvePath(const std::string& path);

struct Options {
  std::string model      = "v49_int8.onnx";
  std::string emb        = "speaker_emb.npy";
  float       power      = 2.0f;     // mask sharpening
  float       gain_db    = 2.0f;     // output gain compensation
  float       lock_db    = kDefaultLockDb;
  int         threads    = 4;
  bool        norm_peak  = false;    // --norm peak
};

class VoiceLock {
 public:
  VoiceLock(const Options& opt, const std::vector<float>& embedding);

  // Processes one chunk. `in` may be any length; internally it is padded or
  // truncated to chunk_samples() because the exported model has a static
  // shape. `out` ends up the same length as `in`.
  ChunkStats Process(const float* in, int len, std::vector<float>* out);

  int   chunk_samples() const { return chunk_samples_; }
  int   hop_samples()   const { return hop_samples_; }
  int   frames()        const { return frames_; }
  float chunk_sec()     const;

  const std::vector<float>& fade_window() const { return fade_; }

  int    processed() const { return processed_; }
  double total_ms()  const { return total_ms_; }
  double rtf()       const;

  void ResetStats();

 private:
  void InspectIo();

  Options opt_;
  float   out_gain_ = 1.0f;

  Ort::Env                     env_;
  Ort::SessionOptions          sess_opt_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo              mem_info_;

  std::vector<std::string> in_names_;
  std::vector<const char*> in_names_c_;

  Stft  stft_;
  int   chunk_samples_ = 0;
  int   hop_samples_   = 0;
  int   frames_        = 0;
  bool  fixed_shape_   = false;

  std::vector<float> spk_emb_;
  std::vector<float> fade_;

  // Reused buffers - avoids allocating inside the realtime loop
  std::vector<float> scratch_in_;
  std::vector<float> scratch_norm_;

  int    processed_ = 0;
  double total_ms_  = 0.0;
};

// Loads an embedding from .npy, checks it is 512-d, and L2-normalizes it.
std::vector<float> LoadEmbedding(const std::string& path);

}  // namespace tse
