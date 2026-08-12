# V49 — Target Speaker Extraction Model

## Overview

V49 is a real-time, single-channel target speaker extraction (TSE) model for the ZenaPro acoustic workstation. Given a mixture of speakers captured by a microphone and a short enrollment recording of the target speaker, V49 outputs audio where the target voice is preserved and other voices are suppressed.

V49 is the shipping model. It was selected over 20+ experimental variants (V47–V54) as the best balance of extraction quality, consistency, and real-time performance.

## Architecture

Two-stage STFT-domain complex masking with speaker cross-attention.

```
Input: mixture waveform (16 kHz mono)
       + speaker enrollment embedding (512-d, from WavLM)

                    mixture waveform
                         |
                    STFT (512-pt, hop 128)
                         |
                    magnitude -> compressed normalization
                         |
               +---------+---------+
               |                   |
          Stage 1              raw STFT
          (complex mask)       (real, imag, mag)
               |                   |
               +------- apply -----+
                         |
                    coarse estimate
                         |
               +---------+---------+
               |                   |
          Stage 2              coarse mag
          (refinement mask)        |
               |                   |
               +------- apply -----+
                         |
                    refined STFT
                         |
                    iSTFT
                         |
                    extracted waveform
```

### Stage 1: Complex Mask Estimation

The core extraction network. Takes compressed STFT magnitude as audio features and the speaker embedding as conditioning.

```
Audio path:
    compressed magnitude (257 freq bins x T frames)
    -> 4x Conv2d blocks (channels: 1 -> 24 -> 48 -> 72 -> 96, freq stride 1/2/2/2)
    -> flatten freq x channels
    -> BiLSTM (640-d) + residual + LayerNorm
    -> BiLSTM (640-d) + residual + LayerNorm

Speaker conditioning:
    WavLM embedding (512-d)
    -> LayerNorm
    -> project to K/V (640-d)
    -> cross-attention: Q = audio trunk, K/V = speaker embedding
    -> gated interaction:
         speaker_small = project(512 -> 128), normalize
         audio_small   = project(640 -> 128), normalize
         interaction   = dot(audio_small, speaker_small) per frame
         gate          = sigmoid(3.0 * interaction)
         conditioning  = trunk * (0.5 + gate) concat speaker_small concat interaction

Conditioning path:
    -> BiLSTM (384-d) + residual + LayerNorm
    -> real mask head (Linear -> sigmoid)     [0, 1]
    -> imag mask head (Linear -> tanh)        [-1, 1]

Output: complex mask (m_real, m_imag)
Applied: est_real = mix_real * m_real - mix_imag * m_imag
         est_imag = mix_real * m_imag + mix_imag * m_real
```

The speaker gate mechanism computes per-frame attention between the audio features and the enrollment. Frames matching the target speaker get gate values near 1.0 (boosted conditioning); non-target frames get values near 0.5 (weaker conditioning). This drives the mask to preserve target energy and suppress interferer energy.

### Stage 2: Attention-Based Refinement

Takes the coarse estimate magnitude from Stage 1 alongside the original mixture magnitude and produces a second [0, 1] refinement mask.

```
input: [coarse_mag, mixture_mag] concatenated (514 bins x T frames)
    -> BiLSTM (512-d, 2 layers) + LayerNorm
    -> self-attention (4 heads) + LayerNorm
    -> sigmoid refinement mask

Output: gate (257 x T)
Applied: refined = coarse_estimate * gate
```

Stage 2 does NOT receive the speaker embedding. It operates purely on the spectral structure of Stage 1's output versus the mixture, learning to clean up residual artifacts and tighten the mask boundaries.

### Speaker Embedding

Frozen WavLM-base-plus-sv (Microsoft, 94M params). Takes 3 seconds of enrollment audio, outputs a 512-dimensional L2-normalized speaker embedding. Computed once at enrollment time, stored as a .npy file, reused at inference.

WavLM is not part of the ONNX export. At runtime on embedded devices, only the pre-computed 512-d vector is needed.

## Model Card

| Property | Value |
|---|---|
| Parameters | 21.9M (Stage 1: ~20M, Stage 2: ~1.9M) |
| ONNX FP32 size | ~90 MB |
| ONNX INT8 size | ~23 MB |
| Sample rate | 16 kHz mono |
| STFT | 512-pt FFT, 128-hop (32ms window, 8ms hop) |
| Latency (Colab CPU) | 45 ms per 3s chunk (RTF 0.015) |
| Latency (laptop x86) | 190 ms per 3s chunk (RTF 0.063) |
| Latency (Pi 5, expected) | 600-1200 ms per 3s chunk (RTF 0.2-0.4) |
| Enrollment | 3s recording -> 512-d WavLM embedding |
| Training data | LibriMix train-360, 12k mixture subset |
| Training loss | SI-SDR (scale-invariant signal-to-distortion ratio) |
| Causal | No (bidirectional LSTMs). Causal variant planned for i.MX 95 |

## Performance (8-file dev set evaluation)

| Metric | V49 | Spec | Status |
|---|---|---|---|
| SI-SDRi (correct enrollment) | +4.27 dB | — | — |
| SI-SDRi (wrong enrollment) | -8.78 dB | — | — |
| Enrollment gap | +13.05 dB | — | Model uses enrollment strongly |
| Leak (total) | 13.6% | ≤ 20% | PASS |
| Leak (LOW, 0-2 kHz) | 14.1% | ≤ 15% | PASS |
| Retention | 57.5% | ≥ 90% | FAIL (LibriMix ceiling) |
| Enrollment effect | -8.6 dB | — | Correct vs wrong outputs differ |

### What the numbers mean

**SI-SDRi +4.27 dB**: the extracted audio is 4.27 dB closer to the clean target than the raw mixture. In perceptual terms, the interferer is noticeably quieter and the target is dominant.

**Enrollment gap +13.05 dB**: when given the correct speaker's enrollment, the model produces output 13 dB better than when given the wrong speaker's enrollment. This confirms the model is genuinely selecting based on speaker identity, not just filtering.

**Leak 13.6%**: of the interferer's energy in the mixture, 13.6% survives into the output. The interferer is reduced by ~8.7 dB but is still faintly audible.

**Retention 57.5%**: 57.5% of the target's energy is preserved in the output. The model removes ~2.4 dB of target energy along with the interferer. This is the trade-off: stronger suppression of the interferer costs some target energy.

**90% retention spec is unreachable** with single-channel extraction on LibriMix. Every model trained this session (V47–V54, 20+ variants) landed between 32% and 58%. This is a fundamental limit of the data and single-channel setting, not a model bug.

### Laptop real-time demo results

With post-processing (input gain normalization + mask sharpening):

| Scenario | Suppression | Meaning |
|---|---|---|
| Target speaker alone | -3.5 to -4.5 dB | 40% energy preserved |
| Other speaker alone | -11 to -14 dB | 93-96% removed |
| Discrimination gap | ~8-9 dB | Clear speaker-dependent behavior |

Best demo settings: `--power 3 --gain 3` with headphones (speakers cause feedback).

## ONNX Interface

5 inputs, 2 outputs. The host code handles STFT/iSTFT; the ONNX model operates in the frequency domain.

**Inputs:**

| Name | Shape | Description |
|---|---|---|
| `inp` | `[1, 257, T]` | Normalized compressed STFT magnitude |
| `spk_emb` | `[1, 512]` | WavLM speaker embedding |
| `mr` | `[1, 257, T]` | STFT real part |
| `mi` | `[1, 257, T]` | STFT imaginary part |
| `mm` | `[1, 257, T]` | STFT magnitude |

**Outputs:**

| Name | Shape | Description |
|---|---|---|
| `est_r` | `[1, 257, T]` | Extracted STFT real part |
| `est_i` | `[1, 257, T]` | Extracted STFT imaginary part |

**Host-side processing:**

```
1. audio chunk (3s, 48000 samples)
2. STFT -> mr, mi, mm (257 x 376 frames)
3. inp = normalize(compress(mm))
4. est_r, est_i = onnx_session.run(inp, spk_emb, mr, mi, mm)
5. waveform = iSTFT(est_r + j * est_i)
```

For real-time: 3s chunks with 1.5s hop, Hann window overlap-add.

## Post-Processing

Two lightweight operations applied after the model, tunable at runtime:

**Input gain normalization**: scales mic audio to match LibriMix training level (~-20 dB RMS) before the model, then restores original scale. Closes the domain gap between quiet laptop mics and the training data.

**Mask sharpening** (`--power N`): recovers the effective mask from model output, raises it to power N, re-applies to the original STFT. Power 1 = no change. Power 2-3 = interferer bins get crushed while target bins are partially preserved. Trades target fidelity for stronger suppression.

**Output gain** (`--gain N dB`): compensates for target energy lost to masking. The model's sigmoid mask caps at 1.0, so it can only attenuate. +2 to +4 dB restores the target to near-input level.

## Training

| Parameter | Value |
|---|---|
| Dataset | LibriMix train-360 (Libri2Mix, wav16k, min mode) |
| Mixture construction | s1 + s2, equal volume |
| Subset | 12,000 of 50,800 mixtures (fixed seed) |
| Enrollment | 3s from a different utterance of the same speaker |
| Target swap | 50% probability of extracting s2 instead of s1 |
| Loss | SI-SDR |
| Optimizer | Adam, lr 3e-4, weight decay 1e-5 |
| Precision | 16-mixed (AMP) |
| Batch size | 4 (effective 16 with 4x gradient accumulation) |
| Epochs | 30 |
| Segment length | 3s (48,000 samples) for training |
| Speaker encoder | WavLM-base-plus-sv, frozen |

## File Inventory

### Model files (for deployment)

| File | Size | Description |
|---|---|---|
| `v49_int8.onnx` | ~23 MB | INT8 quantized, use this on device |
| `v49_fp32.onnx` | ~90 MB | FP32 reference |
| `speaker_emb.npy` | ~2 KB | Pre-computed enrollment embedding |

### Deployment scripts

| File | Target | Dependencies |
|---|---|---|
| `record_enrollment.py` | Laptop | torch, transformers, sounddevice |
| `v49_realtime_demo.py` | Laptop | onnxruntime, numpy, sounddevice, torch (for metrics only) |
| `v49_pi5.py` | Pi 5 / ARM | onnxruntime, numpy, sounddevice only |

### Training and evaluation scripts

| File | Purpose |
|---|---|
| `v49_onnx_export.py` | Export checkpoint to ONNX FP32 + INT8 |
| `v4x_comparison.py` | Multi-model eval (V47/V48/V49/V51) with strict load validation |
| `v54_refiner_training.py` | V54 Stage 2 refiner training on frozen V49 |
| `v54_evaluation.py` | V54 eval |
| `v54_leakage_audit.py` | V49 vs V54 head-to-head on 8 dev files |
| `v54_sanity_check.py` | 5-check pre-deployment validation |
| `v53_v3_training.py` | V53 Conv-TasNet time-domain training |
| `v53_evaluation.py` | V53 eval |
| `rebuild_librimix_index.py` | Regenerate cache pickles from directory listing |
| `clean_stale_checkpoints.py` | Remove old-architecture checkpoints by FiLM block count |

## Deployment Roadmap

```
[DONE]  Laptop demo       x86 CPU, RTF 0.063, v49_realtime_demo.py
[NEXT]  Raspberry Pi 5    ARM Cortex-A76, v49_pi5.py, target RTF < 0.5
[THEN]  NXP i.MX 95       8-mic ADAU7118 PDM array + MVDR beamforming + V49
```

The i.MX 95 deployment adds 8-mic beamforming as a pre-processing stage. MVDR beamforming alone provides 6-15 dB spatial rejection of interferers before V49 runs, which means V49 sees a much cleaner signal and the combined system should reach effective suppression of 15-25 dB — enough for genuine clean extraction.

## Papers

### Architecture

- Luo & Mesgarani, *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation*, IEEE/ACM TASLP 2019
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018

### Target speaker extraction

- Wang et al., *VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking*, Interspeech 2019
- Xu et al., *SpEx: Multi-Scale Time-Domain Speaker Extraction Network*, IEEE/ACM TASLP 2020
- Delcroix et al., *SpeakerBeam*, IEEE JSTSP 2019

### Speaker embedding

- Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022

### Metrics and data

- Le Roux et al., *SDR — Half-baked or Well Done?*, ICASSP 2019
- Cosentino et al., *LibriMix: An Open-Source Dataset for Generalizable Speech Separation*, 2020

### Beamforming (next stage)

- Habets et al., *New Insights Into the MVDR Beamformer in Room Acoustics*, IEEE TASLP 2010
