#!/usr/bin/env python3
"""
VOICE LOCK TEST - V49 Target Speaker Extraction on Raspberry Pi 5
==================================================================

Measures how well V49 "locks" onto a voice: the model keeps the enrolled
speaker and attenuates everyone else. This script reports the attenuation in
dB per chunk and calls each one LOCKED or REJECT in real time.

Install on a Pi 5 - PyTorch is NOT needed for any mode:
    sudo apt install -y libportaudio2
    pip install onnxruntime numpy sounddevice soundfile

Usage
-----
1) Build the enrollment embedding (once). Three options:

   a) On the Pi 5 with the ONNX encoder, no torch (recommended):
        # on a laptop, once: python export_wavlm_onnx.py
        # copy wavlm_sv_int8.onnx to the Pi, then:
        python voice_lock.py enroll --seconds 10

   b) Enroll on a laptop and copy speaker_emb.npy (2 KB) over.

   c) On the Pi with torch (heavy, not recommended):
        pip install torch transformers
        python voice_lock.py enroll --seconds 10

2) Measure speed / RTF on the Pi 5:
     python voice_lock.py bench

3) Realtime test through the mic (WEAR HEADPHONES to avoid feedback):
     python voice_lock.py live --power 3 --gain 3 --save demo1

4) Run the extraction over an existing wav file:
     python voice_lock.py file -i mixed.wav -o extracted.wav

5) A/B the correct voice against a different one:
     python voice_lock.py verify --emb-target target.npy \
         --emb-other other.npy -i mixed.wav

6) Diagnose the embedding itself:
     python voice_lock.py inspect

Tuning knobs:
    --power N   mask sharpening (1 = off, 2 = default, 3 = strong)
    --gain N    output make-up gain in dB (0-4)
    --lock-db N LOCKED decision threshold (default -8 dB)
    --norm peak normalize the output FILE to peak 0.95

ONNX interface (5 in / 2 out); STFT and iSTFT run on the host:
    inp     [1,257,T] compressed, normalized magnitude
    spk_emb [1,512]   WavLM speaker embedding
    mr/mi/mm[1,257,T] STFT real / imag / magnitude
    est_r/est_i [1,257,T]
The model was exported with a STATIC shape: T = 376 -> chunks are exactly 3 s.
"""

import argparse
import os
import re
import sys
import time

import numpy as np


# ============================================================================
# CONFIG - must match the training / export settings exactly
# ============================================================================

SR = 16000
N_FFT = 512
HOP = 128
N_FREQ = N_FFT // 2 + 1          # 257
COMP = 0.3
EPS = 1e-8
SPK_DIM = 512

DEFAULT_CHUNK_SEC = 3.0          # export has a static T -> 3 s
OVERLAP_RATIO = 0.5              # hop = 1.5 s

# Voice lock decision threshold, measured as out/in attenuation in dB
DEFAULT_LOCK_DB = -8.0           # > -8 dB  -> LOCKED (target speaker)
SILENCE_RMS = 3e-3               # below this the chunk counts as silence


# ============================================================================
# STFT / iSTFT - pure numpy, vectorized for ARM
# ============================================================================

def hann_periodic(n):
    """Periodic Hann - same as torch.hann_window(n), NOT np.hanning."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


FFT_WIN = hann_periodic(N_FFT)


def stft(audio):
    """STFT with center=True and reflect padding. Returns (real, imag, mag)."""
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")

    n_frames = 1 + (len(x) - N_FFT) // HOP

    # Build the [T, N_FFT] frame matrix with stride tricks - much faster
    # than a Python loop over frames
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, N_FFT),
        strides=(x.strides[0] * HOP, x.strides[0]),
    )

    spec = np.fft.rfft(frames * FFT_WIN, axis=1)          # [T, F]

    real = np.ascontiguousarray(spec.real.T, dtype=np.float32)
    imag = np.ascontiguousarray(spec.imag.T, dtype=np.float32)
    mag = np.sqrt(real * real + imag * imag + EPS, dtype=np.float32)

    return real, imag, mag


def istft(real, imag, length):
    """Inverse STFT via overlap-add, stripping the center padding."""
    pad = N_FFT // 2
    n_frames = real.shape[1]

    spec = (real + 1j * imag).T                            # [T, F]
    seg = np.fft.irfft(spec, n=N_FFT, axis=1).astype(np.float32) * FFT_WIN

    out_len = (n_frames - 1) * HOP + N_FFT
    output = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    win_sq = FFT_WIN * FFT_WIN

    for t in range(n_frames):
        start = t * HOP
        output[start:start + N_FFT] += seg[t]
        win_sum[start:start + N_FFT] += win_sq

    output /= np.maximum(win_sum, 1e-8)

    return output[pad:pad + length]


def normalize_input(mag):
    """Compressed magnitude normalization - matches V49 training."""
    comp = (mag + EPS) ** COMP
    return ((comp - comp.mean()) / max(comp.std(), 1e-3)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def db(x):
    return 20.0 * np.log10(max(x, 1e-8))


def _mask_verdict(m):
    """Interpret the model's mean mask - the single best diagnostic.

    The mask is the |est| / |mix| ratio the MODEL produced, before it is
    raised to `power`. The final attenuation follows
    20 * power * log10(mask), which has been verified numerically.
    """
    if m > 0.80:
        return "good - model is confident this is the target"
    if m > 0.65:
        return "fair"
    if m > 0.45:
        return "weak - suspect the embedding does not match"
    return "very weak - model does not recognize this voice"


def resolve_device(spec, want_input=True):
    """Translate a device name into a sounddevice index.

    The config file is shared with the C++ build, which talks to ALSA
    directly and therefore uses names like 'plughw:2,0'. PortAudio does not
    understand those, but on Linux its device names usually embed '(hw:2,0)'
    so we match on that.

    Accepts: None | int | '3' | 'plughw:2,0' | 'hw:2,0' | a name fragment
    """
    if spec is None or isinstance(spec, int):
        return spec

    spec = str(spec).strip()

    if spec.isdigit():
        return int(spec)

    if spec in ("default", ""):
        return None

    match = re.match(r"(?:plug)?hw:(\d+)(?:,(\d+))?$", spec)

    if not match:
        return spec           # let sounddevice match it by name

    card = match.group(1)

    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception:
        return None

    # sounddevice uses snake_case (only pyaudio uses camelCase)
    key = "max_input_channels" if want_input else "max_output_channels"
    needle = f"hw:{card},"

    for i, dev in enumerate(devices):
        if needle in dev["name"] and dev[key] > 0:
            print(f"  {spec} -> [{i}] {dev['name']}")
            return i

    print(f"  Warning: could not map '{spec}' to a PortAudio device.")
    print(f"  List them with: python voice_lock.py devices")
    print(f"  Then pass an index instead, e.g. --in-device 3")

    return None


def normalize_output(audio, mode, target_peak=0.95):
    """Normalize the loudness of the output file.

    Applies ONE gain to the whole signal, never per chunk: a per-chunk
    normalizer would lift REJECT segments (other people) back up to LOCKED
    level, destroying the very effect voice lock provides.

    Always call this AFTER suppression has been measured so the figures stay
    honest.
    """
    if mode == "off":
        return audio

    peak = float(np.abs(audio).max())

    if peak < 1e-8:
        return audio

    if mode == "peak":
        return (audio * (target_peak / peak)).astype(np.float32)

    raise ValueError(f"--norm does not support '{mode}'")


# ============================================================================
# EXTRACTOR
# ============================================================================

class VoiceLock:
    """Runs the V49 ONNX model and measures the lock strength per chunk."""

    def __init__(self, model_path, emb, mask_power=2.0, output_gain_db=2.0,
                 lock_db=DEFAULT_LOCK_DB, threads=4):

        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit(
                "onnxruntime is missing. On a Pi 5:\n"
                "    pip install onnxruntime numpy sounddevice soundfile")

        resolved = _resolve(model_path)

        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"  Searched: {', '.join(_SEARCH)}")

        if resolved != model_path:
            print(f"  -> {resolved}")

        model_path = resolved

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1      # Pi 5 has 4 cores, do not oversubscribe
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])

        self._check_io()

        self.spk_emb = self._prepare_emb(emb)

        self.target_rms = 0.1                  # LibriMix RMS level at training
        self.min_rms = SILENCE_RMS
        self.mask_power = float(mask_power)
        self.output_gain = 10.0 ** (float(output_gain_db) / 20.0)
        self.lock_db = float(lock_db)

        # statistics
        self.n_chunks = 0
        self.total_ms = 0.0

    # ------------------------------------------------------------------
    def _check_io(self):
        names = [i.name for i in self.session.get_inputs()]
        expected = ["inp", "spk_emb", "mr", "mi", "mm"]

        missing = [n for n in expected if n not in names]
        if missing:
            raise RuntimeError(
                "Model does not match the V49 interface.\n"
                f"  expected: {expected}\n"
                f"  found   : {names}\n"
                "Is this really a v49_*.onnx file?")

        out_names = [o.name for o in self.session.get_outputs()]
        if "est_r" not in out_names or "est_i" not in out_names:
            raise RuntimeError(f"Unexpected outputs: {out_names}")

        # The export has no dynamic_axes, so T is fixed. Read it from the
        # graph rather than hardcoding it.
        shape = self.session.get_inputs()[names.index("mr")].shape
        t_dim = shape[-1]

        if isinstance(t_dim, int) and t_dim > 0:
            self.n_frames = t_dim
            self.chunk_samples = (t_dim - 1) * HOP
            self.fixed_length = True
        else:
            self.chunk_samples = int(DEFAULT_CHUNK_SEC * SR)
            self.n_frames = self.chunk_samples // HOP + 1
            self.fixed_length = False

        self.chunk_sec = self.chunk_samples / SR
        self.hop_samples = int(self.chunk_samples * OVERLAP_RATIO)
        self.fade_win = hann_periodic(self.chunk_samples)

        print(f"  input T     : {t_dim} frames "
              f"({'fixed' if self.fixed_length else 'dynamic'})")
        print(f"  chunk       : {self.chunk_sec:.2f} s "
              f"({self.chunk_samples} samples), hop "
              f"{self.hop_samples / SR:.2f} s")

    # ------------------------------------------------------------------
    def _prepare_emb(self, emb):
        if isinstance(emb, str):
            path = _resolve(emb)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Embedding not found: {emb}\n"
                    "Create one with: python voice_lock.py enroll --seconds 10")
            emb = np.load(path)

        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)

        if emb.shape[1] != SPK_DIM:
            raise ValueError(
                f"Embedding must be {SPK_DIM}-d, this one is {emb.shape[1]}-d")

        # L2 normalize for safety (WavLM output is already normalized)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    def process(self, chunk):
        """Process one chunk. Returns (extracted, stats dict)."""
        chunk = np.asarray(chunk, dtype=np.float32)

        # Static model shape -> pad or trim to the exact length
        n = len(chunk)
        if n < self.chunk_samples:
            chunk = np.pad(chunk, (0, self.chunk_samples - n))
        elif n > self.chunk_samples:
            chunk = chunk[:self.chunk_samples]

        in_rms = rms(chunk)

        if in_rms < self.min_rms:
            return np.zeros(n, dtype=np.float32), {
                "state": "SILENCE", "in_db": db(in_rms),
                "out_db": -99.0, "suppress_db": 0.0, "ms": 0.0,
                "mask": 0.0,
            }

        # Bring the input level into the training domain
        gain_in = self.target_rms / in_rms
        normalized = chunk * gain_in

        t0 = time.perf_counter()

        mr, mi, mm = stft(normalized)
        mr_b, mi_b, mm_b = mr[None], mi[None], mm[None]
        inp = normalize_input(mm_b)

        est_r, est_i = self.session.run(
            ["est_r", "est_i"],
            {"inp": inp, "spk_emb": self.spk_emb,
             "mr": mr_b, "mi": mi_b, "mm": mm_b},
        )

        # The mask the MODEL produced, before raising it to power. This is
        # the most useful diagnostic available:
        #   ~0.85  the model is confident this is the target
        #   ~0.50  it is unsure -> suspect the embedding
        # Weighted by magnitude, because near-empty bins have a meaningless
        # est/mix ratio and would drag a flat average around.
        est_mag = np.sqrt(est_r ** 2 + est_i ** 2 + EPS)
        mask = np.clip(est_mag / (mm_b + EPS), 0.0, 1.0)

        weight = float(mm_b.sum())
        mask_mean = float((mm_b * mask).sum() / (weight + 1e-12))

        # Mask sharpening: raise the mask to `power`, reapply to the
        # original STFT
        if self.mask_power != 1.0:
            sharp = mask ** self.mask_power
            est_r = mr_b * sharp
            est_i = mi_b * sharp

        extracted = istft(est_r[0], est_i[0], self.chunk_samples)
        extracted = extracted / gain_in * self.output_gain

        ms = (time.perf_counter() - t0) * 1000.0

        self.n_chunks += 1
        self.total_ms += ms

        out_rms = rms(extracted)
        # Exclude output_gain from the measurement so the threshold does not
        # shift when --gain changes
        suppress = db(out_rms / self.output_gain) - db(in_rms)

        return extracted[:n], {
            "state": "LOCKED" if suppress > self.lock_db else "REJECT",
            "in_db": db(in_rms),
            "out_db": db(out_rms),
            "suppress_db": suppress,
            "mask": mask_mean,
            "ms": ms,
        }

    # ------------------------------------------------------------------
    @property
    def rtf(self):
        if self.n_chunks == 0:
            return 0.0
        return (self.total_ms / self.n_chunks) / (self.chunk_sec * 1000.0)


# ============================================================================
# MODE: enroll - build the speaker embedding
# ============================================================================

def cmd_enroll(args):
    import soundfile as sf

    print("=" * 62)
    print("ENROLLMENT - build a 512-d speaker embedding (WavLM)")
    print("=" * 62)

    if args.wav:
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise SystemExit(
                f"File {args.wav} is {sr} Hz, {SR} Hz is required. "
                "Convert first: sox in.wav -r 16000 -c 1 out.wav")
        print(f"  read: {args.wav} ({len(audio) / SR:.1f} s)")
    else:
        import sounddevice as sd

        print(f"\n  Recording {args.seconds:.0f} s. Speak naturally, vary your")
        print("  intonation, read different sentences - the more varied, the")
        print("  more distinctive the embedding.\n")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("  GO\n")

        audio = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1,
                       dtype="float32", device=args.in_device)
        sd.wait()
        audio = audio.squeeze()

        sf.write(args.out_wav, audio, SR)
        print(f"  saved: {args.out_wav}")

    r, peak = rms(audio), float(np.abs(audio).max())
    print(f"  RMS={r:.4f}  peak={peak:.4f}")
    if r < 0.005:
        print("  WARNING: very quiet - check the mic / raise gain (alsamixer)")
    if peak > 0.95:
        print("  WARNING: clipping - move further from the mic")

    encoder = args.encoder or _autodetect_encoder()

    emb, n_seg, consistency = _wavlm_embedding(audio, encoder)

    np.save(args.emb, emb)

    print(f"\n  embedding: {emb.shape} -> {args.emb}")
    print(f"  {n_seg} segments, consistency {consistency:.3f} "
          f"(>0.85 good, <0.7 noisy)")

    if consistency < 0.7:
        print("  Consider re-recording somewhere quieter.")

    print("\n  Try it now:")
    print(f"    python voice_lock.py inspect --emb {args.emb}")
    print(f"    python voice_lock.py live --emb {args.emb}")


# Directories probed when a relative path does not exist. Lets the script run
# from anywhere and still find models/ at the project root.
_SEARCH = ("models", "../models", "../../models", ".", "..", "../..")


def _resolve(path):
    """Locate `path`; if missing, probe its basename inside _SEARCH."""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path

    base = os.path.basename(path)
    for folder in _SEARCH:
        candidate = os.path.join(folder, base)
        if os.path.exists(candidate):
            return candidate

    return path


def _autodetect_encoder():
    """Look for the WavLM ONNX encoder in the usual places."""
    for name in ("wavlm_sv_int8.onnx", "wavlm_sv_fp32.onnx"):
        path = _resolve(name)
        if os.path.exists(path):
            return path
    return None


def _segment(audio):
    """Split into overlapping 3 s segments, dropping the silent ones."""
    seg_len = 3 * SR
    seg_hop = int(1.5 * SR)

    segments = [audio[s:s + seg_len]
                for s in range(0, len(audio) - seg_len + 1, seg_hop)]
    segments = [s for s in segments if rms(s) > 0.005]

    if not segments:
        print("  No segment was loud enough - using the whole recording")
        if len(audio) < seg_len:
            reps = int(np.ceil(seg_len / max(len(audio), 1)))
            audio = np.tile(audio, reps)[:seg_len]
        segments = [audio[:seg_len]]

    return segments


def _norm_segment(seg):
    """Mean/variance normalization - matches WavLM's Wav2Vec2FeatureExtractor."""
    x = np.ascontiguousarray(seg, dtype=np.float32)[None]
    return ((x - x.mean(-1, keepdims=True))
            / np.sqrt(x.var(-1, keepdims=True) + 1e-7)).astype(np.float32)


def _wavlm_embedding(audio, encoder_path=None):
    """10 s -> overlapping 3 s segments -> averaged embedding.

    Prefers the ONNX encoder (no torch required) and falls back to
    transformers + torch when none is available.
    """
    segments = _segment(audio)

    if encoder_path:
        embs = _embed_onnx(segments, encoder_path)
    else:
        embs = _embed_torch(segments)

    avg = np.mean(embs, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)

    if len(embs) > 1:
        cos = [float(np.dot(embs[i], embs[j]))
               for i in range(len(embs)) for j in range(i + 1, len(embs))]
        consistency = float(np.mean(cos))
    else:
        consistency = 1.0

    return avg.astype(np.float32), len(embs), consistency


def _embed_onnx(segments, encoder_path):
    """The main path on a Pi 5 - only needs onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is missing: pip install onnxruntime")

    encoder_path = _resolve(encoder_path)

    if not os.path.exists(encoder_path):
        raise SystemExit(
            f"ONNX encoder not found: {encoder_path}\n"
            "Build it on a laptop with: python export_wavlm_onnx.py")

    print(f"\n  ONNX encoder: {encoder_path} (no PyTorch needed)")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1

    sess = ort.InferenceSession(encoder_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])

    names = [i.name for i in sess.get_inputs()]
    if "input_values" not in names:
        raise SystemExit(
            f"Encoder interface mismatch. Expected 'input_values', got {names}")

    print(f"  {len(segments)} segments", end="", flush=True)

    embs = []
    for seg in segments:
        e = sess.run(["embeddings"], {"input_values": _norm_segment(seg)})[0]
        e = e.ravel().astype(np.float32)
        embs.append(e / (np.linalg.norm(e) + 1e-8))
        print(".", end="", flush=True)

    print(" done")

    if embs[0].shape[0] != SPK_DIM:
        raise SystemExit(
            f"Encoder returned {embs[0].shape[0]}-d, V49 needs {SPK_DIM}-d")

    return embs


def _embed_torch(segments):
    """Fallback when no encoder has been exported - needs torch."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import WavLMForXVector
    except ImportError:
        raise SystemExit(
            "No ONNX encoder and no PyTorch either.\n"
            "Pick one:\n"
            "  a) Enroll on a laptop and copy speaker_emb.npy to the Pi\n"
            "  b) Run export_wavlm_onnx.py on a laptop, copy the .onnx to\n"
            "     the Pi, then enroll with --encoder wavlm_sv_int8.onnx")

    print("\n  Loading WavLM-base-plus-sv (PyTorch)...")
    model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv").eval()

    embs = []
    with torch.no_grad():
        for seg in segments:
            t = torch.from_numpy(_norm_segment(seg))
            e = F.normalize(model(input_values=t).embeddings, dim=-1)
            embs.append(e.squeeze(0).numpy())

    return embs


# ============================================================================
# MODE: inspect - diagnose embedding quality
# ============================================================================

def cmd_inspect(args):
    print("=" * 62)
    print("EMBEDDING ANALYSIS")
    print("=" * 62)

    loaded = []

    for path in args.emb:
        resolved = _resolve(path)

        if not os.path.exists(resolved):
            print(f"\n  {path}: NOT FOUND")
            continue

        raw = np.load(resolved)
        label = os.path.splitext(os.path.basename(resolved))[0]

        print(f"\n  {resolved}")
        print(f"    shape      : {raw.shape}  {raw.dtype}")

        flat = raw.astype(np.float64).ravel()

        n_nan = int(np.isnan(flat).sum())
        n_inf = int(np.isinf(flat).sum())
        norm = float(np.linalg.norm(flat))
        near_zero = int((np.abs(flat) < 1e-6).sum())

        print(f"    L2 norm    : {norm:.4f}"
              f"{'' if abs(norm - 1.0) < 0.01 else '   <-- not a unit vector'}")
        print(f"    mean/std   : {flat.mean():+.5f} / {flat.std():.5f}")
        print(f"    min/max    : {flat.min():+.4f} / {flat.max():+.4f}")
        print(f"    |x| < 1e-6 : {near_zero} / {flat.size}")
        print(f"    NaN / Inf  : {n_nan} / {n_inf}")

        # --- failure modes ---
        problems = []

        if flat.size != SPK_DIM:
            problems.append(
                f"size {flat.size}, V49 needs {SPK_DIM} -> unusable")
        if n_nan or n_inf:
            problems.append("contains NaN/Inf -> corrupt file, re-enroll")
        if norm < 1e-6:
            problems.append(
                "all zeros -> enrollment recorded silence. Check the mic "
                "(--in-device) and the capture level (alsamixer)")
        elif flat.std() < 1e-6:
            problems.append("every value is nearly identical -> meaningless")
        if near_zero > flat.size * 0.5:
            problems.append("over half the values are zero -> likely corrupt")

        if problems:
            for p in problems:
                print(f"    -> ERROR: {p}")
        else:
            print(f"    -> OK")
            loaded.append((label, flat / (norm + 1e-12)))

    # ----------------------------------------------------------------
    # Pairwise comparison
    # ----------------------------------------------------------------
    if len(loaded) > 1:
        print("\n  Pairwise cosine:")
        print()

        width = max(len(n) for n, _ in loaded) + 2
        print("  " + " " * width + "".join(f"{n[:9]:>10}" for n, _ in loaded))

        for name_a, a in loaded:
            row = f"  {name_a:<{width}}"
            for _, b in loaded:
                row += f"{float(np.dot(a, b)):>10.3f}"
            print(row)

    # ----------------------------------------------------------------
    # Compare against a fresh recording - the strongest check there is
    # ----------------------------------------------------------------
    if args.wav:
        import soundfile as sf

        audio, sr = sf.read(_resolve(args.wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"{args.wav} is {sr} Hz, {SR} Hz is required")

        print(f"\n  Computing an embedding from {args.wav} "
              f"({len(audio)/SR:.1f} s)")

        encoder = args.encoder or _autodetect_encoder()
        fresh, n_seg, consistency = _wavlm_embedding(audio, encoder)

        print(f"  {n_seg} segments, consistency {consistency:.3f}")
        print("\n  Cosine against the embeddings above:")

        for name, e in loaded:
            cos = float(np.dot(fresh, e))
            if cos > 0.85:
                verdict = "very good match"
            elif cos > 0.7:
                verdict = "match"
            elif cos > 0.5:
                verdict = "weak - possibly a different person"
            else:
                verdict = "NO match"
            print(f"    {name:<20} {cos:>6.3f}   {verdict}")

    # ----------------------------------------------------------------
    # Reference points
    # ----------------------------------------------------------------
    rng = np.random.default_rng(0)
    r = rng.normal(size=(2000, SPK_DIM))
    r /= np.linalg.norm(r, axis=1, keepdims=True)

    # abs MUST sit outside the sum: we want |dot product|, not sum(|a_i*b_i|)
    cos_random = (r[:1000] * r[1000:]).sum(axis=1)
    baseline = float(np.abs(cos_random).mean())

    print("\n  Reference points:")
    print(f"    two random {SPK_DIM}-d vectors : |cos| ~ {baseline:.3f}"
          f"  (std {cos_random.std():.3f})")
    print("    same person, other recording : usually > 0.85")
    print("    different people             : usually 0.5 - 0.8")
    print("\n  A cosine above 0.95 between two DIFFERENT people means the")
    print("  embedding cannot tell them apart - revisit the enrollment.")


# ============================================================================
# MODE: bench
# ============================================================================

def cmd_bench(args):
    vl = _build(args)

    n = args.iters
    dummy = (np.random.randn(vl.chunk_samples).astype(np.float32) * 0.05)

    print(f"\n  Warmup...")
    for _ in range(3):
        vl.process(dummy)

    vl.n_chunks = 0
    vl.total_ms = 0.0

    times = []
    for _ in range(n):
        _, st = vl.process(dummy)
        times.append(st["ms"])

    times = np.array(times)
    rtf = times.mean() / (vl.chunk_sec * 1000.0)

    print(f"\n  Benchmark ({n} runs, {vl.chunk_sec:.1f} s chunks, "
          f"{args.threads} threads)")
    print(f"    mean   : {times.mean():7.0f} ms")
    print(f"    median : {np.median(times):7.0f} ms")
    print(f"    min/max: {times.min():.0f} / {times.max():.0f} ms")
    print(f"    RTF    : {rtf:.3f}   "
          f"{'REALTIME OK' if rtf < 1.0 else 'TOO SLOW'}")

    budget = vl.hop_samples / SR * 1000.0
    print(f"\n    Budget per hop = {budget:.0f} ms "
          f"-> {'enough' if times.mean() < budget else 'NOT ENOUGH, audio will drop'}")

    if rtf >= 1.0:
        print("\n  Ways to speed this up on a Pi 5:")
        print("    - --power 1 (drops sharpening, ~20% faster)")
        print("    - --threads 4 and close other heavy processes")
        print("    - sudo cpufreq-set -g performance")


# ============================================================================
# MODE: file
# ============================================================================

def cmd_file(args):
    import soundfile as sf

    vl = _build(args)

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"File is {sr} Hz, the model needs {SR} Hz.")

    total = len(audio)
    win, hop = vl.chunk_samples, vl.hop_samples

    if total < win:
        audio = np.pad(audio, (0, win - total))
        total = len(audio)

    output = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)

    print(f"\n  {args.input} ({total / SR:.1f} s)")
    print(f"\n  {'t (s)':>8} {'in dB':>8} {'out dB':>9} "
          f"{'suppress':>10} {'mask':>6} {'ms':>7}  state")
    print("  " + "-" * 66)

    stats = []
    for start in range(0, total - win + 1, hop):
        out, st = vl.process(audio[start:start + win])

        output[start:start + win] += out * vl.fade_win
        weight[start:start + win] += vl.fade_win

        stats.append(st)
        print(f"  {start / SR:8.1f} {st['in_db']:8.1f} {st['out_db']:9.1f} "
              f"{st['suppress_db']:+9.1f} dB {st['mask']:6.3f} {st['ms']:6.0f}"
              f"  {st['state']}")

    output = np.where(weight > 1e-8, output / weight, 0.0)

    peak = float(np.abs(output).max())

    if getattr(args, "norm", "off") != "off":
        # ONE gain for the whole file, applied after suppression has been
        # measured, so the figures stay honest and LOCKED still sits above
        # REJECT.
        output = normalize_output(output, args.norm)
        print(f"\n  (--norm {args.norm}: peak {peak:.4f} -> "
              f"{float(np.abs(output).max()):.4f})")
    elif peak > 1.0:
        output /= peak
        print(f"\n  (rescaled to avoid clipping, peak was {peak:.2f})")

    sf.write(args.output, output, SR)

    _summary(vl, stats)
    print(f"\n  Saved: {args.output}")


# ============================================================================
# MODE: verify - A/B the correct voice against a different one
# ============================================================================

def cmd_verify(args):
    import soundfile as sf

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"File is {sr} Hz, the model needs {SR} Hz.")

    results = {}
    for label, emb_path in (("TARGET", args.emb_target),
                            ("OTHER ", args.emb_other)):
        print(f"\n  --- embedding: {label} ({emb_path}) ---")

        vl = VoiceLock(args.model, emb_path, args.power, args.gain,
                       args.lock_db, args.threads)

        win, hop = vl.chunk_samples, vl.hop_samples
        a = audio if len(audio) >= win else np.pad(audio, (0, win - len(audio)))

        sup = [vl.process(a[s:s + win])[1]["suppress_db"]
               for s in range(0, len(a) - win + 1, hop)]
        sup = [s for s in sup if s != 0.0]

        results[label.strip()] = float(np.mean(sup)) if sup else 0.0
        print(f"      mean suppression: {results[label.strip()]:+.1f} dB")

    gap = results["TARGET"] - results["OTHER"]

    print("\n" + "=" * 62)
    print("VOICE LOCK RESULT")
    print("=" * 62)
    print(f"  correct voice (target) : {results['TARGET']:+6.1f} dB  "
          f"(closer to 0 is better)")
    print(f"  wrong voice   (other)  : {results['OTHER']:+6.1f} dB  "
          f"(more negative is better)")
    print(f"  gap                    : {gap:+6.1f} dB")

    if gap > 6.0:
        print("\n  => GOOD LOCK: the model clearly follows the enrollment.")
    elif gap > 3.0:
        print("\n  => WEAK LOCK: re-enroll for 10-15 s, raise --power.")
    else:
        print("\n  => NO LOCK: check the embedding and the input file.")


# ============================================================================
# MODE: live
# ============================================================================

def cmd_live(args):
    import sounddevice as sd

    vl = _build(args)

    chunk_n = vl.chunk_samples
    hop_n = vl.hop_samples
    fade = vl.fade_win

    state = {
        "inbuf": np.zeros(chunk_n, dtype=np.float32),
        "outbuf": np.zeros(chunk_n, dtype=np.float32),
        "outw": np.zeros(chunk_n, dtype=np.float32),
        "pending": 0,
        "last": {"state": "---", "in_db": -99.0, "out_db": -99.0,
                 "suppress_db": 0.0, "mask": 0.0, "ms": 0.0},
        "hist": [],
    }

    rec_in, rec_out = [], []

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  [audio] {status}", file=sys.stderr)

        x = indata[:, 0].astype(np.float32, copy=True)
        if args.save:
            rec_in.append(x.copy())

        shift = len(x)
        state["inbuf"] = np.roll(state["inbuf"], -shift)
        state["inbuf"][-shift:] = x
        state["pending"] += shift

        if state["pending"] >= hop_n:
            state["pending"] -= hop_n

            extracted, st = vl.process(state["inbuf"].copy())

            state["last"] = st
            state["hist"].append(st)

            state["outbuf"] += extracted * fade
            state["outw"] += fade

        safe = state["outbuf"][:shift]
        w = state["outw"][:shift]
        safe = np.where(w > 1e-8, safe / w, 0.0)
        safe = np.clip(safe, -1.0, 1.0)

        state["outbuf"] = np.roll(state["outbuf"], -shift)
        state["outbuf"][-shift:] = 0.0
        state["outw"] = np.roll(state["outw"], -shift)
        state["outw"][-shift:] = 0.0

        outdata[:, 0] = safe.astype(np.float32)
        if args.save:
            rec_out.append(safe.copy())

    print(f"\n  chunk {vl.chunk_sec:.1f} s / hop {hop_n / SR:.1f} s, "
          f"lock threshold {vl.lock_db:+.1f} dB")
    print("  WEAR HEADPHONES - speakers will cause feedback.")
    if args.save:
        print(f"  recording to: {args.save}_input.wav / {args.save}_output.wav")
    print("  Ctrl+C to stop.\n")
    print(f"  {'chunk':>6} {'in dB':>8} {'out dB':>9} {'suppress':>10} "
          f"{'mask':>6} {'ms':>7} {'RTF':>6}  state")
    print("  " + "-" * 70)

    with sd.Stream(samplerate=SR, blocksize=hop_n, channels=1,
                   dtype="float32", callback=callback, latency="high",
                   device=(args.in_device, args.out_device)):
        try:
            shown = 0
            while True:
                time.sleep(0.5)
                if vl.n_chunks > shown:
                    shown = vl.n_chunks
                    st = state["last"]
                    bar = "#" * max(0, min(20, int(20 + st["suppress_db"])))
                    print(f"  {vl.n_chunks:>6} {st['in_db']:8.1f} "
                          f"{st['out_db']:9.1f} {st['suppress_db']:+9.1f} dB "
                          f"{st['mask']:6.3f} {st['ms']:6.0f} {vl.rtf:6.3f}  "
                          f"{st['state']:<7} {bar}")
        except KeyboardInterrupt:
            print("\n\n  Stopped.")

    _summary(vl, state["hist"])

    if args.save and rec_in:
        import soundfile as sf
        a = np.concatenate(rec_in)
        b = np.concatenate(rec_out)
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]

        if getattr(args, "norm", "off") != "off":
            # Only the saved file is normalized - this changes neither what
            # was played live nor the suppression figures.
            b = normalize_output(b, args.norm)
            print(f"  (--norm {args.norm} applied to the output file)")

        sf.write(f"{args.save}_input.wav", a, SR)
        sf.write(f"{args.save}_output.wav", b, SR)
        print(f"\n  Saved {L / SR:.1f} s:")
        print(f"    {args.save}_input.wav")
        print(f"    {args.save}_output.wav")


# ============================================================================
# HELPERS
# ============================================================================

def _build(args):
    print("=" * 62)
    print("V49 VOICE LOCK")
    print("=" * 62)
    print(f"  model : {args.model}")
    print(f"  emb   : {args.emb}")
    print(f"  power : {args.power}   gain : +{args.gain} dB   "
          f"lock : {args.lock_db:+.1f} dB")

    if getattr(args, "config_used", None):
        print(f"  config: {args.config_used}")

    return VoiceLock(args.model, args.emb, args.power, args.gain,
                     args.lock_db, args.threads)


def _summary(vl, stats):
    stats = [s for s in stats if s["state"] != "SILENCE"]

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)

    if vl.n_chunks == 0 or not stats:
        print("  No chunk contained any sound.")
        return

    sup = np.array([s["suppress_db"] for s in stats])
    locked = sum(1 for s in stats if s["state"] == "LOCKED")

    print(f"  chunks with sound : {len(stats)}")
    print(f"  LOCKED            : {locked} ({100 * locked / len(stats):.0f}%)")
    print(f"  REJECT            : {len(stats) - locked}")
    print(f"  suppression       : mean {sup.mean():+.1f} dB, "
          f"min {sup.min():+.1f} / max {sup.max():+.1f}")

    mk = np.array([s["mask"] for s in stats])
    print(f"  model mask        : mean {mk.mean():.3f}  "
          f"({_mask_verdict(mk.mean())})")
    print(f"  latency           : {vl.total_ms / vl.n_chunks:.0f} ms/chunk, "
          f"RTF {vl.rtf:.3f} "
          f"({'realtime OK' if vl.rtf < 1.0 else 'TOO SLOW'})")


def cmd_devices(_args):
    import sounddevice as sd
    print(sd.query_devices())
    print(f"\n  default: {sd.default.device}")


# ============================================================================
# CONFIG FILE
#
# Format is key = value with '#' comments. Values from the file become the
# argparse DEFAULTS, so command line arguments override them automatically.
# Precedence: defaults  <  voice_lock.conf  <  command line
# ============================================================================

CONFIG_FILE = "voice_lock.conf"

# key -> coercion function
CONFIG_KEYS = {
    "model": str, "emb": str, "encoder": str,
    "power": float, "gain": float, "lock_db": float,
    "threads": int, "iters": int, "seconds": float,
    "in_device": str, "out_device": str,
    "save": str, "input": str, "output": str, "out_wav": str,
    "norm": str,
}


def _load_config(path=None):
    """Read the config file. Returns (values dict, path that was used)."""
    explicit = path is not None
    found = _resolve(path or CONFIG_FILE)

    if not os.path.exists(found):
        if explicit:
            raise SystemExit(f"Config file not found: {path}")
        return {}, None

    cfg = {}

    with open(found, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            if "=" not in line:
                raise SystemExit(f"{found}:{lineno}: missing '=' on line: {line}")

            key, value = line.split("=", 1)
            key = key.strip().replace("-", "_")
            value = value.strip()

            if not value:
                continue                      # present but blank = keep default

            if key not in CONFIG_KEYS:
                # Silently ignoring a misspelled key is a trap: you think you
                # changed the configuration when nothing actually changed.
                valid = "\n".join(f"  {k}" for k in sorted(CONFIG_KEYS))
                raise SystemExit(
                    f"{found}:{lineno}: invalid key '{key}'\n"
                    f"  Valid keys:\n{valid}")

            try:
                cfg[key] = CONFIG_KEYS[key](value)
            except ValueError:
                raise SystemExit(
                    f"{found}:{lineno}: '{value}' is not a valid "
                    f"{CONFIG_KEYS[key].__name__} for key '{key}'")

    return cfg, found


def _prescan_config(argv):
    """Find --config / --no-config before argparse runs."""
    if "--no-config" in argv:
        return None, True

    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 >= len(argv):
            raise SystemExit("Missing path after --config")
        return argv[i + 1], False

    return None, False


# ============================================================================
# CLI
# ============================================================================

def main():
    argv = [a for a in sys.argv[1:]]

    config_path, skip = _prescan_config(argv)
    cfg, cfg_used = ({}, None) if skip else _load_config(config_path)

    # Strip --config/--no-config so argparse never has to know about them
    argv = [a for a in argv if a != "--no-config"]
    if config_path is not None:
        i = argv.index("--config")
        del argv[i:i + 2]

    def d(key, fallback):
        """Default value: from the config file if present, else `fallback`."""
        return cfg.get(key, fallback)

    p = argparse.ArgumentParser(
        description="V49 voice lock test - Raspberry Pi 5",
        epilog="Put the values you use often into voice_lock.conf and stop\n"
               "typing them every time.\n"
               "Precedence: defaults < voice_lock.conf < command line.\n"
               "Skip the file: --no-config    Use another: --config PATH",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_model_args(sp, need_emb=True):
        sp.add_argument("--model", default=d("model", "v49_int8.onnx"),
                        help="the .onnx file (searched under models/)")
        if need_emb:
            sp.add_argument("--emb", default=d("emb", "speaker_emb.npy"),
                            help="512-d speaker embedding (.npy)")
        sp.add_argument("--power", type=float, default=d("power", 2.0),
                        help="mask sharpening (1=off, 2=default, 3=strong)")
        sp.add_argument("--gain", type=float, default=d("gain", 2.0),
                        help="output gain in dB")
        sp.add_argument("--lock-db", type=float,
                        default=d("lock_db", DEFAULT_LOCK_DB),
                        dest="lock_db", help="LOCKED decision threshold (dB)")
        sp.add_argument("--threads", type=int, default=d("threads", 4),
                        help="ONNX thread count (a Pi 5 has 4 cores)")
        sp.add_argument("--norm", choices=("off", "peak"),
                        default=d("norm", "off"),
                        help="normalize the OUTPUT FILE loudness "
                             "(peak = lift to 0.95). Applied after the "
                             "measurement, so suppression figures stay honest.")
        sp.add_argument("--config", metavar="PATH",
                        help="config file (voice_lock.conf is found "
                             "automatically)")
        sp.add_argument("--no-config", action="store_true",
                        help="ignore the config file")

    # enroll
    sp = sub.add_parser("enroll", help="build a speaker embedding")
    sp.add_argument("--encoder", default=d("encoder", None),
                    help="WavLM ONNX encoder (recommended on a Pi 5 - no "
                         "torch needed). Build it with export_wavlm_onnx.py")
    sp.add_argument("--wav", default=None, help="use an existing 16 kHz wav")
    sp.add_argument("--seconds", type=float, default=d("seconds", 10.0))
    sp.add_argument("--emb", default=d("emb", "speaker_emb.npy"))
    sp.add_argument("--out-wav", default=d("out_wav", "enrollment.wav"),
                    dest="out_wav")
    sp.add_argument("--in-device", default=d("in_device", None),
                    dest="in_device")
    sp.add_argument("--config", metavar="PATH")
    sp.add_argument("--no-config", action="store_true")
    sp.set_defaults(func=cmd_enroll)

    # bench
    sp = sub.add_parser("bench", help="measure latency / RTF on a Pi 5")
    add_model_args(sp)
    sp.add_argument("--iters", type=int, default=d("iters", 20))
    sp.set_defaults(func=cmd_bench)

    # file
    sp = sub.add_parser("file", help="extract the target voice from a wav")
    add_model_args(sp)
    sp.add_argument("-i", "--input", default=d("input", None),
                    required="input" not in cfg)
    sp.add_argument("-o", "--output", default=d("output", "extracted.wav"))
    sp.set_defaults(func=cmd_file)

    # verify
    sp = sub.add_parser("verify", help="A/B correct voice vs different voice")
    add_model_args(sp, need_emb=False)
    sp.add_argument("--emb-target", required=True, dest="emb_target")
    sp.add_argument("--emb-other", required=True, dest="emb_other")
    sp.add_argument("-i", "--input", default=d("input", None),
                    required="input" not in cfg)
    sp.set_defaults(func=cmd_verify)

    # live
    sp = sub.add_parser("live", help="realtime test through the mic")
    add_model_args(sp)
    sp.add_argument("--save", default=d("save", None),
                    help="prefix for the recorded input/output wav files")
    sp.add_argument("--in-device", default=d("in_device", None),
                    dest="in_device")
    sp.add_argument("--out-device", default=d("out_device", None),
                    dest="out_device")
    sp.set_defaults(func=cmd_live)

    # inspect
    sp = sub.add_parser("inspect", help="diagnose embedding quality")
    sp.add_argument("--emb", nargs="+", default=[d("emb", "speaker_emb.npy")],
                    help="one or more .npy files to compare")
    sp.add_argument("--wav", default=None,
                    help="a fresh recording to check against the stored "
                         "embedding")
    sp.add_argument("--encoder", default=d("encoder", None))
    sp.add_argument("--config", metavar="PATH")
    sp.add_argument("--no-config", action="store_true")
    sp.set_defaults(func=cmd_inspect)

    # devices
    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(func=cmd_devices)

    args = p.parse_args(argv)
    args.config_used = cfg_used

    # Translate device names (including ALSA-style 'plughw:2,0' coming from
    # the config file shared with the C++ build) into sounddevice indices.
    for name, is_input in (("in_device", True), ("out_device", False)):
        if hasattr(args, name):
            setattr(args, name, resolve_device(getattr(args, name), is_input))

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
