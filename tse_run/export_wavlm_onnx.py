#!/usr/bin/env python3
"""
EXPORT WAVLM SPEAKER ENCODER -> ONNX
=====================================

Chay MOT LAN tren laptop (hoac Colab) de bo hoan toan PyTorch khoi Pi 5.
Sau khi co file .onnx, Pi 5 co the tu enroll ma khong can torch/transformers.

Cai dat (chi tren may export):
    pip install torch transformers onnx onnxruntime

Chay:
    python export_wavlm_onnx.py                    # tao ca fp32 + int8
    python export_wavlm_onnx.py --no-int8          # chi fp32
    python export_wavlm_onnx.py --outdir models

Ket qua:
    wavlm_sv_fp32.onnx   (~380 MB)
    wavlm_sv_int8.onnx   (~95 MB)  <- copy file nay sang Pi 5

Interface:
    input : input_values [1, N]  float32, audio 16 kHz da chuan hoa mean/var
    output: embeddings   [1, 512] float32 (chua L2-normalize)

Truc thoi gian la dynamic, nhung enroll luon dung segment 3 s (48000 samples).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
SEG_SAMPLES = 3 * SR
EMB_DIM = 512
MODEL_ID = "microsoft/wavlm-base-plus-sv"


class XVectorWrapper:
    """Bao WavLMForXVector de forward() tra thang tensor embeddings."""

    def __new__(cls, model):
        import torch.nn as nn

        class _Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_values):
                return self.m(input_values=input_values).embeddings

        return _Wrap(model)


def export(outdir, do_int8, opset):
    try:
        import torch
        from transformers import WavLMForXVector
    except ImportError:
        raise SystemExit(
            "Can: pip install torch transformers onnx onnxruntime\n"
            "Chay script nay tren laptop, KHONG phai tren Pi 5.")

    import onnx
    import onnxruntime as ort

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fp32_path = outdir / "wavlm_sv_fp32.onnx"
    int8_path = outdir / "wavlm_sv_int8.onnx"

    print("=" * 62)
    print("EXPORT WAVLM SPEAKER ENCODER -> ONNX")
    print("=" * 62)

    # ------------------------------------------------------------------
    print(f"\n[1] Load {MODEL_ID}")
    base = WavLMForXVector.from_pretrained(MODEL_ID).eval()
    model = XVectorWrapper(base).eval()

    n_params = sum(p.numel() for p in base.parameters())
    print(f"    params: {n_params:,} ({n_params * 4 / 1e6:.0f} MB fp32)")

    # ------------------------------------------------------------------
    print(f"\n[2] Export fp32 (opset {opset}, truc thoi gian dynamic)")
    torch.manual_seed(0)
    dummy = torch.randn(1, SEG_SAMPLES)

    torch.onnx.export(
        model, (dummy,), str(fp32_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["embeddings"],
        dynamic_axes={"input_values": {1: "n_samples"}},
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(fp32_path)))
    print(f"    {fp32_path.name}  ({fp32_path.stat().st_size / 1e6:.0f} MB)")

    # ------------------------------------------------------------------
    print("\n[3] Parity check PyTorch vs ONNX fp32")
    with torch.no_grad():
        ref = model(dummy).numpy()

    sess = ort.InferenceSession(str(fp32_path),
                                providers=["CPUExecutionProvider"])
    got = sess.run(["embeddings"], {"input_values": dummy.numpy()})[0]

    cos = _cos(ref, got)
    print(f"    max diff : {np.abs(ref - got).max():.2e}")
    print(f"    cosine   : {cos:.6f}  "
          f"{'OK' if cos > 0.9999 else 'CANH BAO — lech qua nhieu'}")

    # ------------------------------------------------------------------
    if do_int8:
        print("\n[4] Quantize INT8 (dynamic)")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(fp32_path), str(int8_path),
                         weight_type=QuantType.QUInt8)

        print(f"    {int8_path.name}  ({int8_path.stat().st_size / 1e6:.0f} MB)")

        s8 = ort.InferenceSession(str(int8_path),
                                  providers=["CPUExecutionProvider"])
        got8 = s8.run(["embeddings"], {"input_values": dummy.numpy()})[0]

        cos8 = _cos(ref, got8)
        print(f"    cosine vs fp32: {cos8:.4f}")

        if cos8 < 0.98:
            print("    CANH BAO: INT8 lech nhieu — nen dung ban fp32 cho enroll.")
            print("    Enroll chi chay 1 lan nen fp32 khong ton kem gi.")
        else:
            print("    INT8 dung duoc cho enroll.")

    # ------------------------------------------------------------------
    print("\n[5] Latency (1 segment 3 s)")
    feed = {"input_values": dummy.numpy()}
    for _ in range(2):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, feed)
    print(f"    fp32: {(time.perf_counter() - t0) * 1000 / 5:.0f} ms "
          f"(tren may nay; Pi 5 cham hon ~5-10x)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("XONG")
    print("=" * 62)
    best = int8_path if do_int8 else fp32_path
    print(f"\n  Copy sang Pi 5:  {best.name}")
    print("\n  Tren Pi 5, enroll khong can torch:")
    print(f"    python voice_lock.py enroll --seconds 10 \\")
    print(f"        --encoder {best.name} --emb speaker_emb.npy")


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser(description="Export WavLM x-vector sang ONNX")
    p.add_argument("--outdir", default="models")
    p.add_argument("--no-int8", action="store_true", dest="no_int8")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    export(args.outdir, not args.no_int8, args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
