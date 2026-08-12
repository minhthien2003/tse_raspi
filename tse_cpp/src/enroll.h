// enroll.h — tao speaker embedding 512-d bang WavLM da export ONNX.
// Khong can PyTorch tren thiet bi.
#pragma once

#include <string>
#include <vector>

namespace tse {

struct EnrollResult {
  std::vector<float> embedding;    // 512-d, da L2-normalize
  int                segments = 0;
  float              consistency = 0.0f;   // cosine trung binh giua cac segment
};

// `encoder_path` la wavlm_sv_int8.onnx / wavlm_sv_fp32.onnx,
// tao bang export_wavlm_onnx.py tren laptop.
EnrollResult ComputeEmbedding(const std::string& encoder_path,
                              const std::vector<float>& audio, int threads);

}  // namespace tse
