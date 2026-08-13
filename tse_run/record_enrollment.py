"""
RECORD ENROLLMENT
==================

Records 3 seconds of your voice through the laptop mic,
computes the WavLM speaker embedding, and saves it as a .npy file.

Run this ONCE before the real-time demo.

Usage:
    python record_enrollment.py

Produces:
    enrollment.wav      (3s recording, for reference)
    speaker_emb.npy     (512-d embedding, used by the demo)

Requirements:
    pip install sounddevice numpy soundfile transformers torch
"""

import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import WavLMForXVector

SR = 16000
DURATION = 3
EMB_PATH = "speaker_emb.npy"
WAV_PATH = "enrollment.wav"


def record():
    print(f"Recording {DURATION}s enrollment...")
    print("Speak naturally - read a sentence aloud.")
    print()

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO")

    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio.squeeze()

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.abs(audio).max()
    print(f"\n  recorded: {len(audio)/SR:.1f}s, RMS={rms:.4f}, peak={peak:.4f}")

    if rms < 0.005:
        print("  WARNING: very quiet - check your mic")
    if peak > 0.95:
        print("  WARNING: clipping - move back from mic")

    sf.write(WAV_PATH, audio, SR)
    print(f"  saved: {WAV_PATH}")
    return audio


def compute_embedding(audio):
    print("\nComputing speaker embedding (WavLM)...")
    wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    with torch.no_grad():
        t = torch.from_numpy(audio).float().unsqueeze(0)
        mean = t.mean(dim=-1, keepdim=True)
        var = t.var(dim=-1, keepdim=True)
        normalized = (t - mean) / torch.sqrt(var + 1e-7)
        emb = wavlm(input_values=normalized).embeddings
        emb = F.normalize(emb, dim=-1)

    emb_np = emb.squeeze(0).numpy()
    np.save(EMB_PATH, emb_np)
    print(f"  embedding shape: {emb_np.shape}")
    print(f"  saved: {EMB_PATH}")
    return emb_np


def main():
    print("=" * 50)
    print("ENROLLMENT RECORDING")
    print("=" * 50)

    audio = record()
    compute_embedding(audio)

    print()
    print("Done. Now run the demo:")
    print(f"  python v49_realtime_demo.py --model v49_int8.onnx --emb {EMB_PATH}")


if __name__ == "__main__":
    main()
