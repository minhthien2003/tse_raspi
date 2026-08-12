#!/usr/bin/env python3
"""
PARITY REFERENCE — so khop DSP giua ban Python va ban C++
==========================================================

Sinh dung tin hieu ma `tse_voice_lock selftest` dung, roi in cac con so
tuong ung. Hai ben phai khop toi ~1e-5. Neu lech, KHONG chay model —
STFT sai thi output chi la rac.

Chay:
    python tools/parity_ref.py                 # in so cua ban Python
    ./build/tse_voice_lock selftest            # in so cua ban C++
    # so bang mat, hoac:
    python tools/parity_ref.py --check ./build/tse_voice_lock

Chi can numpy — khong can onnxruntime.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 16000
N_FFT = 512
HOP = 128
N_FREQ = N_FFT // 2 + 1
COMP = 0.3
EPS = 1e-8

# Sai so cho phep giua hai ban. FFT cua hai ben dung thuat toan khac nhau
# (numpy pocketfft vs radix-2 tu viet) nen khong the bang tuyet doi.
TOL_REL = 2e-5
TOL_ABS = 1e-5

# Cac dai luong ky vong bang 0 phai so TUYET DOI. So tuong doi voi 0
# luon that bai du hai ben deu dung.
ABSOLUTE = {"roundtrip", "norm_mean"}


def hann_periodic(n):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


FFT_WIN = hann_periodic(N_FFT)


def stft(audio):
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP

    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, N_FFT),
        strides=(x.strides[0] * HOP, x.strides[0]))

    spec = np.fft.rfft(frames * FFT_WIN, axis=1)

    real = np.ascontiguousarray(spec.real.T, dtype=np.float32)
    imag = np.ascontiguousarray(spec.imag.T, dtype=np.float32)
    mag = np.sqrt(real * real + imag * imag + EPS, dtype=np.float32)

    return real, imag, mag


def istft(real, imag, length):
    pad = N_FFT // 2
    n_frames = real.shape[1]

    spec = (real + 1j * imag).T
    seg = np.fft.irfft(spec, n=N_FFT, axis=1).astype(np.float32) * FFT_WIN

    out_len = (n_frames - 1) * HOP + N_FFT
    output = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    win_sq = FFT_WIN * FFT_WIN

    for t in range(n_frames):
        s = t * HOP
        output[s:s + N_FFT] += seg[t]
        win_sum[s:s + N_FFT] += win_sq

    output /= np.maximum(win_sum, 1e-8)
    return output[pad:pad + length]


def normalize_input(mag):
    comp = (mag + EPS) ** COMP
    return ((comp - comp.mean()) / max(comp.std(), 1e-3)).astype(np.float32)


def norm_mean_var(x):
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


def test_signal():
    """Phai giong het ham sinh tin hieu trong CmdSelftest cua main.cpp."""
    n = 48000
    t = np.arange(n, dtype=np.float64) / SR
    x = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 1234.0 * t)
    return x.astype(np.float32)


def compute():
    x = test_signal()

    real, imag, mag = stft(x)
    y = istft(real, imag, len(x))

    norm = normalize_input(mag)
    mv = norm_mean_var(x)

    return {
        "frames": float(mag.shape[1]),
        "roundtrip": float(np.abs(x - y).max()),
        "sum_mag": float(mag.sum()),
        "mag_14_50": float(mag[14, 50]),
        "norm_mean": float(norm.mean()),
        "meanvar_1000": float(mv[1000]),
    }


LABELS = {
    "frames":       "frames",
    "roundtrip":    "roundtrip err",
    "sum_mag":      "sum(mag)",
    "mag_14_50":    "mag[14,50]",
    "norm_mean":    "norm mean",
    "meanvar_1000": "meanvar[1000]",
}


def parse_cpp(text):
    """Boc so tu output cua `tse_voice_lock selftest`."""
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "frames":       grab(r"frames\s*:\s*(\d+)"),
        "roundtrip":    grab(r"roundtrip err\s*:\s*([0-9.eE+-]+)"),
        "sum_mag":      grab(r"sum\(mag\)\s*:\s*([0-9.eE+-]+)"),
        "mag_14_50":    grab(r"mag\[14,50\]\s*:\s*([0-9.eE+-]+)"),
        "norm_mean":    grab(r"norm mean\s*:\s*([0-9.eE+-]+)"),
        "meanvar_1000": grab(r"meanvar\[1000\]\s*:\s*([0-9.eE+-]+)"),
    }


def main():
    p = argparse.ArgumentParser(description="Doi chieu DSP Python vs C++")
    p.add_argument("--check", metavar="BINARY", default=None,
                   help="chay binary C++ va so sanh tu dong")
    args = p.parse_args()

    py = compute()

    print("=" * 62)
    print("PARITY REFERENCE (Python)")
    print("=" * 62)
    for k, v in py.items():
        print(f"  {LABELS[k]:<16}: {v:.6g}")

    if py["frames"] != 376:
        print("\n  LOI: frames phai la 376 — cau hinh STFT sai")
        return 1

    if py["roundtrip"] > 1e-5:
        print("\n  LOI: roundtrip Python da sai, sua truoc khi so C++")
        return 1

    if not args.check:
        print("\n  Chay ban C++ roi so bang mat:")
        print("    ./build/tse_voice_lock selftest")
        print("\n  Hoac tu dong:")
        print("    python tools/parity_ref.py --check ./build/tse_voice_lock")
        return 0

    binary = Path(args.check)
    if not binary.exists():
        print(f"\n  Khong tim thay binary: {binary}")
        return 1

    out = subprocess.run([str(binary), "selftest"],
                         capture_output=True, text=True).stdout
    cpp = parse_cpp(out)

    print("\n" + "=" * 62)
    print("SO SANH")
    print("=" * 62)
    print(f"  {'gia tri':<16} {'Python':>14} {'C++':>14} {'lech':>10}")
    print(f"  {'':<16} {'':>14} {'':>14} {'(abs/rel)':>10}")
    print("  " + "-" * 58)

    ok = True
    for k in py:
        c = cpp.get(k)
        if c is None:
            print(f"  {LABELS[k]:<16} {'':>14} {'THIEU':>14}")
            ok = False
            continue

        diff = abs(py[k] - c)

        if k in ABSOLUTE:
            good = diff < TOL_ABS
            shown = diff
        else:
            shown = diff / max(abs(py[k]), 1e-30)
            good = shown < TOL_REL

        ok &= good

        print(f"  {LABELS[k]:<16} {py[k]:>14.6g} {c:>14.6g} "
              f"{shown:>9.2e} {'' if good else '  <-- LECH'}")

    print("\n  " + ("PASS — hai ban khop, C++ dung duoc"
                    if ok else "FAIL — dung chay model, sua DSP truoc"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
