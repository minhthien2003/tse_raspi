#include "enroll.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "core.h"

namespace tse {

namespace {

// Cat thanh segment 3 s chong lan 1.5 s, bo doan im lang.
std::vector<std::vector<float>> Segment(const std::vector<float>& audio) {
  std::vector<std::vector<float>> segs;

  const int n = static_cast<int>(audio.size());

  for (int s = 0; s + kEnrollSegment <= n; s += kEnrollHop) {
    const float* p = audio.data() + s;
    if (Rms(p, kEnrollSegment) > 0.005f) {
      segs.emplace_back(p, p + kEnrollSegment);
    }
  }

  if (segs.empty()) {
    std::printf("  Khong co segment nao du to — dung toan bo audio\n");

    std::vector<float> pad = audio;
    while (static_cast<int>(pad.size()) < kEnrollSegment && !audio.empty()) {
      pad.insert(pad.end(), audio.begin(), audio.end());
    }
    pad.resize(kEnrollSegment, 0.0f);
    segs.push_back(std::move(pad));
  }

  return segs;
}

}  // namespace

EnrollResult ComputeEmbedding(const std::string& encoder_path,
                              const std::vector<float>& audio, int threads) {
  const std::vector<std::vector<float>> segs = Segment(audio);

  std::printf("\n  Encoder ONNX: %s\n", encoder_path.c_str());

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "enroll");

  Ort::SessionOptions opt;
  opt.SetIntraOpNumThreads(threads);
  opt.SetInterOpNumThreads(1);
  opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  Ort::Session session(env, encoder_path.c_str(), opt);

  Ort::AllocatorWithDefaultOptions alloc;
  const std::string in_name(session.GetInputNameAllocated(0, alloc).get());

  if (in_name != "input_values") {
    throw std::runtime_error(
        "Encoder khong dung interface. Can input 'input_values', thay '" +
        in_name + "'.\n  Tao lai bang: python export_wavlm_onnx.py");
  }

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const char* in_names[]  = {"input_values"};
  const char* out_names[] = {"embeddings"};

  std::vector<std::vector<float>> embs;
  embs.reserve(segs.size());

  std::printf("  %zu segment", segs.size());
  std::fflush(stdout);

  std::vector<float> norm;

  for (const auto& seg : segs) {
    NormalizeMeanVar(seg.data(), static_cast<int>(seg.size()), norm);

    const int64_t dims[2] = {1, static_cast<int64_t>(norm.size())};

    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, norm.data(), norm.size(), dims, 2);

    std::vector<Ort::Value> out = session.Run(
        Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);

    const size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();

    if (static_cast<int>(n) != kSpkDim) {
      throw std::runtime_error(
          "Encoder tra ve " + std::to_string(n) + "-d, V49 can " +
          std::to_string(kSpkDim) + "-d");
    }

    const float* p = out[0].GetTensorData<float>();
    std::vector<float> e(p, p + n);

    double norm2 = 0.0;
    for (float v : e) norm2 += static_cast<double>(v) * v;
    norm2 = std::sqrt(norm2) + 1e-8;
    for (float& v : e) v = static_cast<float>(v / norm2);

    embs.push_back(std::move(e));

    std::printf(".");
    std::fflush(stdout);
  }

  std::printf(" xong\n");

  // Trung binh roi chuan hoa lai
  std::vector<float> avg(kSpkDim, 0.0f);
  for (const auto& e : embs) {
    for (int i = 0; i < kSpkDim; ++i) avg[i] += e[i];
  }

  double norm2 = 0.0;
  for (float& v : avg) {
    v /= static_cast<float>(embs.size());
    norm2 += static_cast<double>(v) * v;
  }
  norm2 = std::sqrt(norm2) + 1e-8;
  for (float& v : avg) v = static_cast<float>(v / norm2);

  // Consistency = cosine trung binh giua moi cap segment
  float consistency = 1.0f;

  if (embs.size() > 1) {
    double sum = 0.0;
    int pairs = 0;

    for (size_t i = 0; i < embs.size(); ++i) {
      for (size_t j = i + 1; j < embs.size(); ++j) {
        double dot = 0.0;
        for (int k = 0; k < kSpkDim; ++k) dot += embs[i][k] * embs[j][k];
        sum += dot;
        ++pairs;
      }
    }

    consistency = static_cast<float>(sum / pairs);
  }

  EnrollResult r;
  r.embedding    = std::move(avg);
  r.segments     = static_cast<int>(embs.size());
  r.consistency  = consistency;
  return r;
}

}  // namespace tse
