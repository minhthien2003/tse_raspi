// enroll.h - builds a 512-d speaker embedding using WavLM exported to ONNX.
// No PyTorch is needed on the device.
#pragma once

#include <string>
#include <vector>

namespace tse {

struct EnrollResult {
  std::vector<float> embedding;      // 512-d, L2-normalized
  int                segments = 0;
  float              consistency = 0.0f;   // mean cosine across segments
};

// `encoder_path` points at wavlm_sv_int8.onnx / wavlm_sv_fp32.onnx, produced
// by export_wavlm_onnx.py on a laptop.
EnrollResult ComputeEmbedding(const std::string& encoder_path,
                              const std::vector<float>& audio, int threads);

}  // namespace tse
