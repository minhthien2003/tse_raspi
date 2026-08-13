"""
V49 SPEAKER EXTRACTION — Raspberry Pi 5 Deployment
====================================================

Minimal runtime for ARM deployment. No PyTorch, no transformers at runtime.

Dependencies (install on Pi 5):
    pip install onnxruntime numpy sounddevice soundfile

Files needed on Pi 5:
    v49_int8.onnx      (~23 MB, the model)
    speaker_emb.npy    (~2 KB, pre-computed on laptop via record_enrollment.py)
    v49_pi5.py         (this script)

Usage:
    # Real-time extraction
    python v49_pi5.py --model v49_int8.onnx --emb speaker_emb.npy

    # Real-time with recording
    python v49_pi5.py --model v49_int8.onnx --emb speaker_emb.npy --save demo1

    # Process a file
    python v49_pi5.py --model v49_int8.onnx --emb speaker_emb.npy --file input.wav -o output.wav

    # Tune extraction strength
    python v49_pi5.py --model v49_int8.onnx --emb speaker_emb.npy --power 3 --gain 3

Tuning:
    --power N   mask sharpening (1=off, 2=default, 3=aggressive, 4=max)
    --gain N    output gain in dB (compensates target attenuation, 0-4)

Performance target:
    Pi 5 (Cortex-A76 @ 2.4 GHz): RTF ~0.2-0.4 expected
    Laptop (x86):                 RTF ~0.06 measured
    RTF < 1.0 = real-time capable

Notes:
    - Enrollment (speaker_emb.npy) must be pre-computed on a machine with
      PyTorch + transformers. The Pi 5 never runs WavLM.
    - Use headphones during real-time mode to prevent audio feedback.
    - STFT is computed in pure numpy (no torch dependency).
"""

import argparse
import sys
import os
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd


# ============================================================================
# CONFIG
# ============================================================================

SR = 16000
N_FFT = 512
HOP = 128
N_FREQ = N_FFT // 2 + 1   # 257
COMP = 0.3
EPS = 1e-8

CHUNK_SEC = 3.0
HOP_SEC = 1.5
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)


# ============================================================================
# STFT (pure numpy, no torch)
# ============================================================================

FFT_WIN = np.hanning(N_FFT).astype(np.float32)
FADE_WIN = np.hanning(CHUNK_SAMPLES).astype(np.float32)


def stft(audio):
    """Real STFT. Returns (real, imag, magnitude) as float32 arrays."""
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP

    real = np.zeros((N_FREQ, n_frames), dtype=np.float32)
    imag = np.zeros((N_FREQ, n_frames), dtype=np.float32)

    for t in range(n_frames):
        start = t * HOP
        spec = np.fft.rfft(x[start:start + N_FFT] * FFT_WIN)
        real[:, t] = spec.real.astype(np.float32)
        imag[:, t] = spec.imag.astype(np.float32)

    mag = np.sqrt(real ** 2 + imag ** 2 + EPS)
    return real, imag, mag


def istft(real, imag, length):
    """Inverse STFT. Returns waveform of given length."""
    pad = N_FFT // 2
    n_frames = real.shape[1]
    out_len = (n_frames - 1) * HOP + N_FFT
    output = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)

    for t in range(n_frames):
        spec = real[:, t] + 1j * imag[:, t]
        seg = np.fft.irfft(spec, n=N_FFT).astype(np.float32) * FFT_WIN
        start = t * HOP
        output[start:start + N_FFT] += seg
        win_sum[start:start + N_FFT] += FFT_WIN ** 2

    output = output / np.maximum(win_sum, 1e-8)
    return output[pad:pad + length]


def normalize_input(mag):
    """Compressed magnitude normalization matching V49 training."""
    comp = (mag + EPS) ** COMP
    mean = comp.mean()
    std = max(comp.std(), 1e-3)
    return (comp - mean) / std


# ============================================================================
# EXTRACTOR
# ============================================================================

class V49Extractor:
    def __init__(self, model_path, emb_path, mask_power=2.0, output_gain_db=2.0):
        # ONNX session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Pre-computed speaker embedding
        self.spk_emb = np.load(emb_path).reshape(1, -1).astype(np.float32)

        # Processing params
        self.target_rms = 0.1
        self.min_rms = 1e-6
        self.mask_power = mask_power
        self.output_gain = 10 ** (output_gain_db / 20)

    def process(self, audio_chunk):
        """Process one chunk. Returns extracted waveform (same length)."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms < self.min_rms:
            return np.zeros_like(audio_chunk)

        gain_in = self.target_rms / rms
        normalized = audio_chunk * gain_in

        mr, mi, mm = stft(normalized)
        mr_b = mr[np.newaxis]
        mi_b = mi[np.newaxis]
        mm_b = mm[np.newaxis]
        inp = normalize_input(mm_b)

        est_r, est_i = self.session.run(
            ["est_r", "est_i"],
            {"inp": inp, "spk_emb": self.spk_emb,
             "mr": mr_b, "mi": mi_b, "mm": mm_b},
        )

        if self.mask_power != 1.0:
            est_mag = np.sqrt(est_r ** 2 + est_i ** 2 + EPS)
            mask = np.clip(est_mag / (mm_b + EPS), 0, 1)

            # Frequency-dependent sharpening (must match laptop script).
            # Same-gender overlap worst below 1250 Hz. Sharpen harder there.
            power_map = np.ones((1, N_FREQ, 1), dtype=np.float32)
            power_map[:, :40, :] = self.mask_power + 1.0   # 0-1250 Hz extra
            power_map[:, 40:80, :] = self.mask_power + 0.5  # 1250-2500 Hz
            power_map[:, 80:, :] = self.mask_power           # 2500+ Hz base

            sharp = np.power(mask, power_map)
            est_r = mr_b * sharp
            est_i = mi_b * sharp

        extracted = istft(est_r[0], est_i[0], len(audio_chunk))
        return extracted / gain_in * self.output_gain


# ============================================================================
# REAL-TIME STREAM
# ============================================================================

class RealtimeStream:
    def __init__(self, extractor, save_prefix=None):
        self.extractor = extractor
        self.input_buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        self.output_buffer = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        self.output_weight = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        self.samples_received = 0
        self.chunks_processed = 0
        self.total_latency_ms = 0

        self.last_in_rms = 0
        self.last_out_rms = 0
        self.last_suppress = 0

        self.save_prefix = save_prefix
        self.rec_input = []
        self.rec_output = []

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"  status: {status}", file=sys.stderr)

        audio_in = indata[:, 0].copy()
        self.rec_input.append(audio_in.copy())

        shift = len(audio_in)
        self.input_buffer = np.roll(self.input_buffer, -shift)
        self.input_buffer[-shift:] = audio_in
        self.samples_received += shift

        if self.samples_received >= HOP_SAMPLES:
            self.samples_received -= HOP_SAMPLES

            t0 = time.perf_counter()
            extracted = self.extractor.process(self.input_buffer.copy())
            ms = (time.perf_counter() - t0) * 1000

            self.chunks_processed += 1
            self.total_latency_ms += ms

            self.last_in_rms = np.sqrt(np.mean(self.input_buffer ** 2))
            self.last_out_rms = np.sqrt(np.mean(extracted ** 2))
            if self.last_in_rms > 1e-6:
                self.last_suppress = 20 * np.log10(
                    self.last_out_rms / (self.last_in_rms + 1e-8))

            self.output_buffer += extracted * FADE_WIN
            self.output_weight += FADE_WIN

        safe = self.output_buffer[:shift].copy()
        w = self.output_weight[:shift].copy()
        safe = np.where(w > 1e-8, safe / w, 0.0)

        self.output_buffer = np.roll(self.output_buffer, -shift)
        self.output_buffer[-shift:] = 0.0
        self.output_weight = np.roll(self.output_weight, -shift)
        self.output_weight[-shift:] = 0.0

        outdata[:, 0] = safe.astype(np.float32)
        self.rec_output.append(safe.copy())

    def run(self):
        print(f"\n  chunk: {CHUNK_SEC}s, hop: {HOP_SEC}s")
        if self.save_prefix:
            print(f"  recording to: {self.save_prefix}_input.wav / _output.wav")
        print(f"  Ctrl+C to stop\n")
        print(f"  {'chunks':>7} {'latency':>9} {'RTF':>7} {'in_dB':>8} {'out_dB':>8} {'suppress':>10}")
        print(f"  {'-'*55}")

        blocksize = int(HOP_SEC * SR)

        with sd.Stream(
            samplerate=SR, blocksize=blocksize, channels=1,
            dtype="float32", callback=self.callback, latency="low",
        ):
            try:
                while True:
                    time.sleep(1)
                    if self.chunks_processed > 0:
                        avg = self.total_latency_ms / self.chunks_processed
                        rtf = avg / (CHUNK_SEC * 1000)
                        in_db = 20 * np.log10(self.last_in_rms + 1e-8)
                        out_db = 20 * np.log10(self.last_out_rms + 1e-8)
                        print(f"  {self.chunks_processed:>7} "
                              f"{avg:>7.0f}ms "
                              f"{rtf:>7.3f} "
                              f"{in_db:>+7.1f}dB "
                              f"{out_db:>+7.1f}dB "
                              f"{self.last_suppress:>+9.1f}dB")
            except KeyboardInterrupt:
                pass

        # Summary
        print("\n\n  Stopped.")
        if self.chunks_processed > 0:
            avg = self.total_latency_ms / self.chunks_processed
            rtf = avg / (CHUNK_SEC * 1000)
            print(f"  chunks: {self.chunks_processed}, "
                  f"avg latency: {avg:.0f} ms, RTF: {rtf:.3f}")
            if rtf < 1.0:
                print(f"  real-time: YES (RTF {rtf:.3f})")
            else:
                print(f"  real-time: NO (RTF {rtf:.3f}, need RTF < 1.0)")
                print(f"  try: reduce --power or increase chunk size")

        # Save
        if self.save_prefix and self.rec_input:
            import soundfile as sf
            inp = np.concatenate(self.rec_input)
            out = np.concatenate(self.rec_output)
            L = min(len(inp), len(out))
            inp_path = f"{self.save_prefix}_input.wav"
            out_path = f"{self.save_prefix}_output.wav"
            sf.write(inp_path, inp[:L], SR)
            sf.write(out_path, out[:L], SR)
            print(f"\n  Saved ({L/SR:.1f}s):")
            print(f"    {inp_path}")
            print(f"    {out_path}")


# ============================================================================
# FILE MODE
# ============================================================================

def process_file(extractor, input_path, output_path):
    import soundfile as sf

    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        print(f"  WARNING: file is {sr} Hz, model expects {SR} Hz")

    total = len(audio)
    win = CHUNK_SAMPLES
    hop = HOP_SAMPLES
    output = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)

    n = 0
    total_ms = 0
    for start in range(0, total - win + 1, hop):
        t0 = time.perf_counter()
        out = extractor.process(audio[start:start + win])
        total_ms += (time.perf_counter() - t0) * 1000
        output[start:start + win] += out * FADE_WIN
        weight[start:start + win] += FADE_WIN
        n += 1

    output = np.where(weight > 1e-8, output / weight, 0.0)

    sf.write(output_path, output, SR)
    avg = total_ms / max(n, 1)
    rtf = avg / (CHUNK_SEC * 1000)
    print(f"  {input_path} ({total/SR:.1f}s) -> {output_path}")
    print(f"  {n} chunks, avg {avg:.0f} ms, RTF {rtf:.3f}")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark(extractor, n_iters=20):
    """Measure pure inference latency without audio I/O."""
    dummy = np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.01

    # Warmup
    for _ in range(3):
        extractor.process(dummy)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        extractor.process(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    rtf = times.mean() / (CHUNK_SEC * 1000)
    print(f"\n  Benchmark ({n_iters} iterations, {CHUNK_SEC}s chunks):")
    print(f"    mean:   {times.mean():.0f} ms")
    print(f"    median: {np.median(times):.0f} ms")
    print(f"    min:    {times.min():.0f} ms")
    print(f"    max:    {times.max():.0f} ms")
    print(f"    RTF:    {rtf:.3f}  {'(real-time OK)' if rtf < 1 else '(TOO SLOW)'}")

    if rtf >= 1.0:
        print(f"\n  Not real-time on this hardware.")
        print(f"  Options:")
        print(f"    - Use --power 1 (no mask sharpening, saves ~20% compute)")
        print(f"    - Increase chunk size to reduce STFT overhead ratio")
        print(f"    - Use FP32 model on hardware with better FP support")

    return rtf


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V49 Speaker Extraction (Pi 5 / ARM deployment)")
    parser.add_argument("--model", required=True, help="v49_int8.onnx")
    parser.add_argument("--emb", required=True, help="speaker_emb.npy")
    parser.add_argument("--power", type=float, default=2.0,
                        help="Mask sharpening (1=off, 2=default, 3=aggressive)")
    parser.add_argument("--gain", type=float, default=2.0,
                        help="Output gain dB (0-4)")
    parser.add_argument("--save", default=None,
                        help="Save prefix for input/output WAV recording")
    parser.add_argument("--file", default=None,
                        help="Process a file instead of live mic")
    parser.add_argument("-o", "--output", default="extracted.wav",
                        help="Output path for file mode")
    parser.add_argument("--bench", action="store_true",
                        help="Run latency benchmark and exit")
    args = parser.parse_args()

    print("=" * 50)
    print("V49 SPEAKER EXTRACTION")
    print(f"  model: {args.model}")
    print(f"  power: {args.power}  gain: +{args.gain} dB")
    print("=" * 50)

    extractor = V49Extractor(
        args.model, args.emb,
        mask_power=args.power, output_gain_db=args.gain,
    )

    if args.bench:
        benchmark(extractor)
    elif args.file:
        process_file(extractor, args.file, args.output)
    else:
        stream = RealtimeStream(extractor, save_prefix=args.save)
        stream.run()


if __name__ == "__main__":
    main()
