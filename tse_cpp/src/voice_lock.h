// voice_lock.h — chay V49 ONNX va do muc khoa giong.
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
  double    ms          = 0.0;
};

// Tim file trong cac vi tri thong thuong: nguyen van, roi models/ o thu muc
// hien tai, cha, va ong. Cho phep chay binary tu tse_cpp/build/ ma van thay
// models/ o goc project.
// Tra ve chuoi rong neu khong tim thay o dau.
std::string ResolvePath(const std::string& path);

struct Options {
  std::string model      = "v49_int8.onnx";
  std::string emb        = "speaker_emb.npy";
  float       power      = 2.0f;     // mask sharpening
  float       gain_db    = 2.0f;     // bu gain dau ra
  float       lock_db    = kDefaultLockDb;
  int         threads    = 4;
};

class VoiceLock {
 public:
  VoiceLock(const Options& opt, const std::vector<float>& embedding);

  // Chay mot chunk. `in` dai bat ky; ben trong se pad/cat cho khop
  // chunk_samples() vi model export co shape tinh. `out` co cung do dai `in`.
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

  // Bo dem tai su dung — tranh cap phat trong vong lap realtime
  std::vector<float> scratch_in_;
  std::vector<float> scratch_norm_;

  int    processed_ = 0;
  double total_ms_  = 0.0;
};

// Doc embedding tu .npy, kiem tra 512-d va L2-normalize.
std::vector<float> LoadEmbedding(const std::string& path);

}  // namespace tse
