"""
V49 ONNX EXPORT
=================

Exports V49 (best model, verified clean load) to ONNX for laptop real-time demo.

Pipeline on laptop:
    1. Record 3s enrollment -> WavLM -> speaker embedding (done once)
    2. Stream mic audio in chunks
    3. Per chunk: STFT -> normalize -> V49 ONNX -> iSTFT -> speaker audio out

Produces:
    v49_fp32.onnx   (~90 MB)
    v49_int8.onnx   (~23 MB, use this on laptop)

Interface (5 inputs, 2 outputs):
    Inputs:
        inp      [1, 257, T]   normalized compressed magnitude
        spk_emb  [1, 512]      WavLM speaker embedding
        mr       [1, 257, T]   STFT real
        mi       [1, 257, T]   STFT imaginary
        mm       [1, 257, T]   STFT magnitude
    Outputs:
        est_r    [1, 257, T]   extracted STFT real
        est_i    [1, 257, T]   extracted STFT imaginary

Host code does iSTFT(est_r + j*est_i) to get the waveform.
"""

import os
import sys
import re
import math
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONFIG
# ============================================================================

PROJECT = Path("/content/drive/MyDrive/zena_tse_training")
V49_CKPT_DIR = PROJECT / "exp/tse_v49_stage2_attention/checkpoints"
OUT_DIR = PROJECT / "exp/v49_onnx_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP32_PATH = OUT_DIR / "v49_fp32.onnx"
INT8_PATH = OUT_DIR / "v49_int8.onnx"

DEVICE = "cpu"  # export from CPU

SR = 16000
N_FFT = 512
HOP = 128
N_FREQ = N_FFT // 2 + 1
COMP = 0.3
EPS = 1e-8

CLIP_SEC = 3
CLIP_SAMPLES = CLIP_SEC * SR
T_FRAMES = CLIP_SAMPLES // HOP + 1

# Architecture
CONV_CH = [24, 48, 72, 96]
LSTM_HIDDEN, LSTM_OUT = 320, 640
SPK_DIM, SPK_REDUCED = 512, 128
COND_HIDDEN, COND_OUT = 192, 384
STAGE2_EMB, STAGE2_HEADS = 512, 4


# ============================================================================
# DEPENDENCIES
# ============================================================================

def ensure_packages():
    for pkg in ["onnx", "onnxruntime"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

ensure_packages()
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


# ============================================================================
# V49 ARCHITECTURE
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=(stride, 1), padding=1)
        self.norm = nn.GroupNorm(1, out_c)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Stage1Masker(nn.Module):
    def __init__(self):
        super().__init__()
        strides = [1, 2, 2, 2]
        self.conv_blocks = nn.ModuleList([
            ConvBlock(1 if i == 0 else CONV_CH[i-1], CONV_CH[i], strides[i])
            for i in range(4)
        ])
        flat = CONV_CH[-1] * math.ceil(math.ceil(math.ceil(math.ceil(N_FREQ) / 2) / 2) / 2)

        self.lstm1 = nn.LSTM(flat, LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.residual1 = nn.Linear(flat, LSTM_OUT)
        self.norm1 = nn.LayerNorm(LSTM_OUT)

        self.lstm2 = nn.LSTM(LSTM_OUT, LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(LSTM_OUT)

        self.speaker_norm = nn.LayerNorm(SPK_DIM)
        self.speaker_projection = nn.Linear(SPK_DIM, SPK_REDUCED)
        self.audio_query = nn.Linear(LSTM_OUT, SPK_REDUCED)
        self.speaker_to_kv = nn.Linear(SPK_DIM, LSTM_OUT)
        self.cross_attn = nn.MultiheadAttention(LSTM_OUT, 4, batch_first=True)
        self.attn_norm = nn.LayerNorm(LSTM_OUT)

        self.condition_lstm = nn.LSTM(LSTM_OUT + SPK_REDUCED + 1, COND_HIDDEN,
                                      batch_first=True, bidirectional=True)
        self.condition_residual = nn.Linear(LSTM_OUT, COND_OUT)
        self.condition_norm = nn.LayerNorm(COND_OUT)

        self.real_head = nn.Linear(COND_OUT, N_FREQ)
        self.imag_head = nn.Linear(COND_OUT, N_FREQ)

    def forward(self, feat, spk):
        h = feat.unsqueeze(1)
        for block in self.conv_blocks:
            h = block(h)
        h = h.permute(0, 3, 1, 2).contiguous()
        b, t, c, f = h.shape
        h = h.view(b, t, c * f)

        o, _ = self.lstm1(h)
        first = self.norm1(o + self.residual1(h))
        o, _ = self.lstm2(first)
        trunk = self.norm2(o + first)

        spk_norm = self.speaker_norm(spk)
        spk_kv = self.speaker_to_kv(spk_norm).unsqueeze(1)
        attn_out, _ = self.cross_attn(trunk, spk_kv, spk_kv)
        cond_trunk = self.attn_norm(trunk + attn_out)

        spk_small = F.normalize(self.speaker_projection(spk_norm), dim=-1)
        aud_small = F.normalize(self.audio_query(cond_trunk), dim=-1)
        interaction = (aud_small * spk_small.unsqueeze(1)).sum(-1, keepdim=True)
        gate = torch.sigmoid(3.0 * interaction)

        cond_input = torch.cat([
            cond_trunk * (0.5 + gate),
            spk_small.unsqueeze(1).expand(-1, t, -1),
            interaction,
        ], dim=-1)
        cond_h, _ = self.condition_lstm(cond_input)
        cond_h = self.condition_norm(cond_h + self.condition_residual(cond_trunk))

        m_real = torch.sigmoid(self.real_head(cond_h)).transpose(1, 2)
        m_imag = torch.tanh(self.imag_head(cond_h)).transpose(1, 2)
        return m_real, m_imag


class V49AttnStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FREQ * 2, 256, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(STAGE2_EMB)
        self.self_attn = nn.MultiheadAttention(STAGE2_EMB, STAGE2_HEADS,
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(STAGE2_EMB)
        self.refine_mask = nn.Linear(STAGE2_EMB, N_FREQ)

    def forward(self, mix_mag, stage1_mag, spk=None):
        x = torch.cat([mix_mag, stage1_mag], dim=1).transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.norm1(out)
        attn_out, _ = self.self_attn(out, out, out)
        out = self.norm2(out + attn_out)
        return torch.sigmoid(self.refine_mask(out)).transpose(1, 2)


class V49Model(nn.Module):
    """Returns real/imag as separate tensors (ONNX-friendly, no complex)."""

    def __init__(self):
        super().__init__()
        self.stage1 = Stage1Masker()
        self.stage2 = V49AttnStage2()

    def forward(self, inp, spk_emb, mr, mi, mm):
        m_real, m_imag = self.stage1(inp, spk_emb)
        est1_r = mr * m_real - mi * m_imag
        est1_i = mr * m_imag + mi * m_real
        est1_mag = torch.sqrt(est1_r * est1_r + est1_i * est1_i + EPS)
        gate = self.stage2(mm, est1_mag)
        return est1_r * gate, est1_i * gate


# ============================================================================
# LOADING
# ============================================================================

def find_best_checkpoint():
    names = sorted(os.listdir(V49_CKPT_DIR))
    ckpts = [V49_CKPT_DIR / n for n in names
             if n.endswith(".ckpt") and n != "last.ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {V49_CKPT_DIR}")
    def score(c):
        m = re.search(r"val_sisdri=([+-]?\d+\.\d+)", c.name)
        return float(m.group(1)) if m else -1e9
    return max(ckpts, key=score)


def load_model(ckpt_path):
    model = V49Model().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    clean = {k.replace("model.", "", 1): v for k, v in sd.items()
             if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"V49 load mismatch: {len(missing)} missing, {len(unexpected)} unexpected\n"
            f"  missing: {list(missing)[:6]}\n"
            f"  unexpected: {list(unexpected)[:6]}"
        )
    return model


# ============================================================================
# EXPORT
# ============================================================================

def make_dummy_inputs():
    torch.manual_seed(42)
    wav = torch.randn(1, CLIP_SAMPLES)
    window = torch.hann_window(N_FFT)
    stft = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=window, return_complex=True)
    mr, mi, mm = stft.real, stft.imag, stft.abs()
    comp = (mm + EPS).pow(COMP)
    inp = (comp - comp.mean()) / comp.std().clamp(min=1e-3)
    spk = torch.randn(1, SPK_DIM)
    return inp, spk, mr, mi, mm


def export_fp32(model, dummy, path):
    inp, spk, mr, mi, mm = dummy
    torch.onnx.export(
        model, (inp, spk, mr, mi, mm), str(path),
        export_params=True, opset_version=17,
        do_constant_folding=True,
        input_names=["inp", "spk_emb", "mr", "mi", "mm"],
        output_names=["est_r", "est_i"],
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def quantize(fp32_path, int8_path):
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)


def verify_parity(model, dummy, onnx_path):
    inp, spk, mr, mi, mm = dummy
    with torch.no_grad():
        ref_r, ref_i = model(inp, spk, mr, mi, mm)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_r, onnx_i = sess.run(
        ["est_r", "est_i"],
        {"inp": inp.numpy(), "spk_emb": spk.numpy(),
         "mr": mr.numpy(), "mi": mi.numpy(), "mm": mm.numpy()},
    )
    diff = np.abs(ref_r.numpy() - onnx_r)
    rel = float(diff.mean() / (np.abs(ref_r.numpy()).mean() + 1e-8) * 100)
    return {"max_diff": float(diff.max()), "mean_diff": float(diff.mean()), "rel_pct": rel}


def benchmark(onnx_path, dummy, n_iters=10):
    inp, spk, mr, mi, mm = dummy
    feed = {"inp": inp.numpy(), "spk_emb": spk.numpy(),
            "mr": mr.numpy(), "mi": mi.numpy(), "mm": mm.numpy()}
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for _ in range(3): sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(n_iters): sess.run(None, feed)
    ms = (time.perf_counter() - t0) * 1000 / n_iters
    rtf = ms / (CLIP_SAMPLES / SR * 1000)
    return {"latency_ms": ms, "rtf": rtf}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("V49 ONNX EXPORT")
    print("=" * 60)

    # 1. Load
    ckpt = find_best_checkpoint()
    print(f"\n[1] Checkpoint: {ckpt.name}")
    model = load_model(ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params: {n_params:,} ({n_params * 4 / 1e6:.1f} MB FP32)")
    print(f"    loaded clean (0 missing, 0 unexpected)")

    # 2. Dummy inputs
    print(f"\n[2] Test inputs ({CLIP_SEC}s clip, {T_FRAMES} STFT frames)")
    dummy = make_dummy_inputs()
    print(f"    inp:     {tuple(dummy[0].shape)}")
    print(f"    spk_emb: {tuple(dummy[1].shape)}")
    print(f"    mr/mi/mm:{tuple(dummy[2].shape)}")

    # 3. FP32
    print(f"\n[3] Exporting FP32...")
    export_fp32(model, dummy, FP32_PATH)
    fp32_mb = os.path.getsize(FP32_PATH) / 1e6
    print(f"    saved: {FP32_PATH.name} ({fp32_mb:.1f} MB)")

    # 4. Parity
    print(f"\n[4] Parity check (PyTorch vs ONNX FP32)")
    p = verify_parity(model, dummy, FP32_PATH)
    print(f"    max diff:  {p['max_diff']:.2e}")
    print(f"    mean diff: {p['mean_diff']:.2e}")
    print(f"    relative:  {p['rel_pct']:.2f}%")

    # 5. INT8
    print(f"\n[5] INT8 quantization...")
    quantize(FP32_PATH, INT8_PATH)
    int8_mb = os.path.getsize(INT8_PATH) / 1e6
    print(f"    saved: {INT8_PATH.name} ({int8_mb:.1f} MB)")

    # INT8 drift
    p8 = verify_parity(model, dummy, INT8_PATH)
    print(f"    INT8 drift: {p8['rel_pct']:.2f}%")

    # 6. Benchmark
    print(f"\n[6] CPU latency ({CLIP_SEC}s chunk)")
    fp32_b = benchmark(FP32_PATH, dummy)
    int8_b = benchmark(INT8_PATH, dummy)
    print(f"    FP32: {fp32_b['latency_ms']:.0f} ms  (RTF={fp32_b['rtf']:.3f})")
    print(f"    INT8: {int8_b['latency_ms']:.0f} ms  (RTF={int8_b['rtf']:.3f})")

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"  FP32: {FP32_PATH}")
    print(f"  INT8: {INT8_PATH}")
    print()
    print(f"  Download {INT8_PATH.name} to laptop (~{int8_mb:.0f} MB)")
    print()
    print("  Laptop usage:")
    print("    1. Record 3s enrollment of target speaker")
    print("    2. Run WavLM on enrollment -> 512-d embedding (once)")
    print("    3. Stream mic in 3s chunks:")
    print("       STFT -> normalize -> v49_int8.onnx -> iSTFT -> output")
    print()
    print("  RTF < 1.0 means real-time capable on CPU.")
    if int8_b['rtf'] < 1.0:
        print(f"  INT8 RTF = {int8_b['rtf']:.3f} -> real-time OK")
    else:
        print(f"  INT8 RTF = {int8_b['rtf']:.3f} -> too slow for real-time on this CPU")
        print(f"  Try on laptop CPU (likely faster than Colab CPU)")


if __name__ == "__main__":
    main()
