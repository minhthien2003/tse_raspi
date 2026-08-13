// io.h - WAV and NPY reading/writing without external libraries.
//
// The NPY side must stay compatible with numpy in both directions: a
// speaker_emb.npy written by np.save on a laptop has to load here, and
// files written here have to load in Python.
#pragma once

#include <string>
#include <vector>

namespace tse {

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

// Reads mono/stereo WAV, PCM 16/24/32-bit or float32. Stereo is downmixed to
// mono. Samples come back in [-1, 1]. Throws std::runtime_error on failure.
// sample_rate receives the file's actual rate (NO resampling is performed).
std::vector<float> ReadWav(const std::string& path, int* sample_rate);

// Writes mono 16-bit PCM WAV.
void WriteWav(const std::string& path, const std::vector<float>& samples,
              int sample_rate);

// ---------------------------------------------------------------------------
// NPY (float32, 1-D or 2-D, C-order)
// ---------------------------------------------------------------------------

std::vector<float> ReadNpyFloat32(const std::string& path);

void WriteNpyFloat32(const std::string& path, const std::vector<float>& data);

}  // namespace tse
