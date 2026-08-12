#include "voice_lock.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "io.h"

namespace tse {

const char* ToString(LockState s) {
  switch (s) {
    case LockState::kLocked:  return "LOCKED";
    case LockState::kReject:  return "REJECT";
    default:                  return "SILENCE";
  }
}

// ===========================================================================
// Tim file
// ===========================================================================

std::string ResolvePath(const std::string& path) {
  if (path.empty()) return "";

  // Duong dan tuyet doi thi khong doan them
  const bool absolute = path[0] == '/' ||
                        (path.size() > 1 && path[1] == ':');

  auto exists = [](const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
  };

  if (exists(path)) return path;
  if (absolute) return "";

  // Chi lay ten file de ghep voi cac thu muc models/
  std::string base = path;
  const size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);

  const char* prefixes[] = {
      "models/", "../models/", "../../models/", "../../../models/",
      "", "../", "../../",
  };

  for (const char* p : prefixes) {
    const std::string candidate = std::string(p) + base;
    if (exists(candidate)) return candidate;
  }

  return "";
}

namespace {

std::string MustResolve(const std::string& path, const char* what) {
  const std::string found = ResolvePath(path);

  if (found.empty()) {
    throw std::runtime_error(
        std::string("Khong tim thay ") + what + ": " + path +
        "\n  Da tim o: ./ , models/ , ../models/ , ../../models/"
        "\n  Chi ro bang duong dan tuyet doi neu de o cho khac.");
  }

  return found;
}

}  // namespace

// ===========================================================================
// Embedding
// ===========================================================================

std::vector<float> LoadEmbedding(const std::string& path) {
  std::vector<float> e = ReadNpyFloat32(MustResolve(path, "embedding"));

  if (static_cast<int>(e.size()) != kSpkDim) {
    throw std::runtime_error(
        "Embedding phai la " + std::to_string(kSpkDim) + "-d, file co " +
        std::to_string(e.size()) + " phan tu: " + path);
  }

  double norm = 0.0;
  for (float v : e) norm += static_cast<double>(v) * v;
  norm = std::sqrt(norm) + 1e-8;

  for (float& v : e) v = static_cast<float>(v / norm);

  return e;
}

// ===========================================================================
// VoiceLock
// ===========================================================================

VoiceLock::VoiceLock(const Options& opt, const std::vector<float>& embedding)
    : opt_(opt),
      env_(ORT_LOGGING_LEVEL_WARNING, "voice_lock"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      spk_emb_(embedding) {
  if (static_cast<int>(spk_emb_.size()) != kSpkDim) {
    throw std::runtime_error("Embedding khong phai 512-d");
  }

  out_gain_ = std::pow(10.0f, opt_.gain_db / 20.0f);

  sess_opt_.SetIntraOpNumThreads(opt_.threads);
  sess_opt_.SetInterOpNumThreads(1);   // Pi 5 chi 4 core, tranh oversubscribe
  sess_opt_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  const std::string model = MustResolve(opt_.model, "model ONNX");
  if (model != opt_.model) std::printf("  -> %s\n", model.c_str());

  session_ = std::make_unique<Ort::Session>(env_, model.c_str(), sess_opt_);

  InspectIo();

  fade_ = HannPeriodic(chunk_samples_);
  scratch_in_.resize(chunk_samples_);
}

void VoiceLock::InspectIo() {
  Ort::AllocatorWithDefaultOptions alloc;

  const size_t n_in = session_->GetInputCount();
  in_names_.reserve(n_in);

  for (size_t i = 0; i < n_in; ++i) {
    in_names_.emplace_back(session_->GetInputNameAllocated(i, alloc).get());
  }
  for (const auto& s : in_names_) in_names_c_.push_back(s.c_str());

  auto has = [this](const char* n) {
    return std::find(in_names_.begin(), in_names_.end(), n) != in_names_.end();
  };

  const char* required[] = {"inp", "spk_emb", "mr", "mi", "mm"};
  for (const char* r : required) {
    if (!has(r)) {
      std::string got;
      for (const auto& s : in_names_) got += s + " ";
      throw std::runtime_error(
          std::string("Model khong dung interface V49. Thieu input '") + r +
          "'.\n  Can : inp spk_emb mr mi mm\n  Thay: " + got);
    }
  }

  // Model export khong co dynamic_axes -> truc thoi gian co dinh.
  // Lay T tu graph thay vi hardcode, phong khi sau nay re-export.
  const size_t mr_idx = static_cast<size_t>(
      std::find(in_names_.begin(), in_names_.end(), "mr") - in_names_.begin());

  const std::vector<int64_t> shape = session_->GetInputTypeInfo(mr_idx)
                                         .GetTensorTypeAndShapeInfo()
                                         .GetShape();

  const int64_t t_dim = shape.empty() ? -1 : shape.back();

  if (t_dim > 0) {
    frames_        = static_cast<int>(t_dim);
    chunk_samples_ = (frames_ - 1) * kHop;
    fixed_shape_   = true;
  } else {
    chunk_samples_ = 3 * kSampleRate;
    frames_        = stft_.FramesFor(chunk_samples_);
    fixed_shape_   = false;
  }

  hop_samples_ = chunk_samples_ / 2;

  std::printf("  input T     : %lld frames (%s)\n",
              static_cast<long long>(t_dim),
              fixed_shape_ ? "co dinh" : "dong");
  std::printf("  chunk       : %.2f s (%d samples), hop %.2f s\n",
              chunk_sec(), chunk_samples_,
              static_cast<double>(hop_samples_) / kSampleRate);
}

float VoiceLock::chunk_sec() const {
  return static_cast<float>(chunk_samples_) / kSampleRate;
}

double VoiceLock::rtf() const {
  if (processed_ == 0) return 0.0;
  return (total_ms_ / processed_) / (chunk_sec() * 1000.0);
}

void VoiceLock::ResetStats() {
  processed_ = 0;
  total_ms_  = 0.0;
}

ChunkStats VoiceLock::Process(const float* in, int len,
                              std::vector<float>* out) {
  ChunkStats st;

  out->assign(len, 0.0f);

  const float in_rms = Rms(in, len);
  st.in_db = Db(in_rms);

  if (in_rms < kSilenceRms) {
    st.state = LockState::kSilence;
    return st;
  }

  const auto t0 = std::chrono::steady_clock::now();

  // Model co shape tinh -> ep ve dung chunk_samples_
  std::fill(scratch_in_.begin(), scratch_in_.end(), 0.0f);
  const int n = std::min(len, chunk_samples_);

  // Chuan hoa muc vao ve ~ -20 dB RMS cho khop domain LibriMix
  const float gain_in = 0.1f / in_rms;
  for (int i = 0; i < n; ++i) scratch_in_[i] = in[i] * gain_in;

  const Spectrogram s = stft_.Forward(scratch_in_.data(), chunk_samples_);

  if (s.frames != frames_) {
    throw std::runtime_error(
        "So frame STFT (" + std::to_string(s.frames) + ") khong khop model (" +
        std::to_string(frames_) + ")");
  }

  NormalizeInput(s.mag, scratch_norm_);

  const int64_t dims[3] = {1, kNFreq, s.frames};
  const size_t  count   = static_cast<size_t>(kNFreq) * s.frames;
  const int64_t emb_dims[2] = {1, kSpkDim};

  // Tensor tro thang vao bo dem cua ta — ORT khong copy.
  auto tensor = [&](std::vector<float>& v) {
    return Ort::Value::CreateTensor<float>(mem_info_, v.data(), count, dims, 3);
  };

  std::vector<float> mr = s.real, mi = s.imag, mm = s.mag;

  std::vector<Ort::Value> inputs;
  inputs.reserve(in_names_.size());

  for (const auto& name : in_names_) {
    if (name == "inp") {
      inputs.push_back(tensor(scratch_norm_));
    } else if (name == "mr") {
      inputs.push_back(tensor(mr));
    } else if (name == "mi") {
      inputs.push_back(tensor(mi));
    } else if (name == "mm") {
      inputs.push_back(tensor(mm));
    } else {  // spk_emb
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info_, spk_emb_.data(), spk_emb_.size(), emb_dims, 2));
    }
  }

  const char* out_names[] = {"est_r", "est_i"};

  std::vector<Ort::Value> outputs = session_->Run(
      Ort::RunOptions{nullptr}, in_names_c_.data(), inputs.data(),
      inputs.size(), out_names, 2);

  float* est_r = outputs[0].GetTensorMutableData<float>();
  float* est_i = outputs[1].GetTensorMutableData<float>();

  // Mask sharpening: lay lai mask hieu dung roi mu len power.
  if (opt_.power != 1.0f) {
    for (size_t i = 0; i < count; ++i) {
      const float em =
          std::sqrt(est_r[i] * est_r[i] + est_i[i] * est_i[i] + kEps);
      float mask = em / (mm[i] + kEps);
      mask = std::max(0.0f, std::min(1.0f, mask));
      mask = std::pow(mask, opt_.power);

      est_r[i] = mr[i] * mask;
      est_i[i] = mi[i] * mask;
    }
  }

  std::vector<float> wave(chunk_samples_);
  stft_.Inverse(est_r, est_i, s.frames, wave.data(), chunk_samples_);

  const float restore = out_gain_ / gain_in;
  for (int i = 0; i < n; ++i) (*out)[i] = wave[i] * restore;

  st.ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t0)
              .count();

  ++processed_;
  total_ms_ += st.ms;

  const float out_rms = Rms(out->data(), len);
  st.out_db = Db(out_rms);

  // Bo out_gain khoi phep do, neu khong --gain se lam lech nguong lock.
  st.suppress_db = Db(out_rms / out_gain_) - st.in_db;
  st.state = st.suppress_db > opt_.lock_db ? LockState::kLocked
                                           : LockState::kReject;

  return st;
}

}  // namespace tse
