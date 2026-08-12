#!/usr/bin/env python3
"""
VOICE LOCK TEST — V49 Target Speaker Extraction on Raspberry Pi 5
==================================================================

Kiem tra kha nang "khoa giong" (voice lock) cua model V49:
model chi giu lai giong cua nguoi da enroll, nen giong khac se bi
suy giam manh. Script do muc suy giam (dB) tren tung chunk va bao
LOCKED / REJECT theo thoi gian thuc.

Cai dat tren Pi 5 — KHONG can PyTorch cho bat ky che do nao:
    sudo apt install -y libportaudio2
    pip install onnxruntime numpy sounddevice soundfile

Cach dung
---------
1) Tao enrollment embedding (chi lam 1 lan). Ba lua chon:

   a) Tren Pi 5 voi encoder ONNX — khong can torch (khuyen dung):
        # tren laptop, 1 lan: python export_wavlm_onnx.py
        # copy wavlm_sv_int8.onnx sang Pi, roi:
        python voice_lock.py enroll --seconds 10

   b) Tren laptop roi copy file speaker_emb.npy (2 KB) sang Pi.

   c) Tren Pi voi torch (nang, khong khuyen khich):
        pip install torch transformers
        python voice_lock.py enroll --seconds 10

2) Do toc do / RTF trên Pi 5:
     python voice_lock.py bench --model v49_int8.onnx --emb speaker_emb.npy

3) Test voice lock realtime qua mic (deo tai nghe de tranh hu):
     python voice_lock.py live --model v49_int8.onnx --emb speaker_emb.npy \
         --power 3 --gain 3 --save demo1

4) Test voice lock tren file wav san co:
     python voice_lock.py file --model v49_int8.onnx --emb speaker_emb.npy \
         -i mixed.wav -o extracted.wav

5) Kiem tra A/B (giong dung vs giong sai) tren 2 file:
     python voice_lock.py verify --model v49_int8.onnx \
         --emb-target target_emb.npy --emb-other other_emb.npy -i mixed.wav

Tham so tinh chinh:
    --power N   sharpening mask (1 = tat, 2 = mac dinh, 3 = manh)
    --gain N    bu gain dau ra theo dB (0-4)
    --lock-db N nguong quyet dinh LOCKED (mac dinh -8 dB)

ONNX interface (5 in / 2 out), STFT & iSTFT chay o host:
    inp     [1,257,T] magnitude nen + chuan hoa
    spk_emb [1,512]   WavLM speaker embedding
    mr/mi/mm[1,257,T] STFT real / imag / magnitude
    est_r/est_i [1,257,T]
Model duoc export voi shape TINH: T = 376 -> chunk bat buoc 3.0 s.
"""

import argparse
import os
import sys
import time

import numpy as np


# ============================================================================
# CONFIG — phai khop chinh xac voi luc training / export
# ============================================================================

SR = 16000
N_FFT = 512
HOP = 128
N_FREQ = N_FFT // 2 + 1          # 257
COMP = 0.3
EPS = 1e-8
SPK_DIM = 512

DEFAULT_CHUNK_SEC = 3.0          # model export co T tinh -> 3 s
OVERLAP_RATIO = 0.5              # hop = 1.5 s

# Nguong quyet dinh voice lock (do bang dB suy giam out/in)
DEFAULT_LOCK_DB = -8.0           # > -8 dB  -> LOCKED (giong target)
SILENCE_RMS = 3e-3               # duoi muc nay coi nhu im lang


# ============================================================================
# STFT / iSTFT — numpy thuan, vector hoa cho ARM
# ============================================================================

def hann_periodic(n):
    """Hann periodic — giong torch.hann_window(n) (KHONG phai np.hanning)."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


FFT_WIN = hann_periodic(N_FFT)


def stft(audio):
    """STFT center=True, pad reflect. Tra ve (real, imag, mag) [F, T]."""
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")

    n_frames = 1 + (len(x) - N_FFT) // HOP

    # Tao ma tran frame [T, N_FFT] bang stride trick -> nhanh hon vong for
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
    """iSTFT overlap-add, bo phan pad center. Tra ve waveform dai `length`."""
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
    """Compressed magnitude normalization — khop voi V49 training."""
    comp = (mag + EPS) ** COMP
    return ((comp - comp.mean()) / max(comp.std(), 1e-3)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def db(x):
    return 20.0 * np.log10(max(x, 1e-8))


def _mask_verdict(m):
    """Doc mask trung binh cua model — chi so chan doan quan trong nhat.

    Mask la ty le |est| / |mix| MA MODEL TAO RA, truoc khi mu len power.
    Suy giam cuoi cung = 20 * power * log10(mask), da kiem chung bang so.
    """
    if m > 0.80:
        return "tot — model tu tin day la giong dich"
    if m > 0.65:
        return "kha"
    if m > 0.45:
        return "yeu — nghi ngo embedding khong khop"
    return "rat yeu — model khong nhan ra giong nay"


def normalize_output(audio, mode, target_peak=0.95):
    """Chuan hoa am luong file dau ra.

    Chi ap dung MOT he so cho toan bo tin hieu, tuyet doi khong chuan hoa
    theo tung chunk: lam vay se keo cac doan REJECT (giong nguoi khac) len
    lai bang doan LOCKED, tuc la pha huy chinh tac dung voice lock.

    Luon goi SAU khi da do suppress, de so do khong bi lech.
    """
    if mode == "off":
        return audio

    peak = float(np.abs(audio).max())

    if peak < 1e-8:
        return audio

    if mode == "peak":
        return (audio * (target_peak / peak)).astype(np.float32)

    raise ValueError(f"--norm khong ho tro '{mode}'")


# ============================================================================
# EXTRACTOR
# ============================================================================

class VoiceLock:
    """Chay V49 ONNX + do muc khoa giong tren tung chunk."""

    def __init__(self, model_path, emb, mask_power=2.0, output_gain_db=2.0,
                 lock_db=DEFAULT_LOCK_DB, threads=4):

        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit(
                "Thieu onnxruntime. Cai tren Pi 5:\n"
                "    pip install onnxruntime numpy sounddevice soundfile")

        resolved = _resolve(model_path)

        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Khong tim thay model: {model_path}\n"
                f"  Da do o: {', '.join(_SEARCH)}")

        if resolved != model_path:
            print(f"  -> {resolved}")

        model_path = resolved

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1          # Pi 5: 4 core, tranh oversubscribe
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])

        self._check_io()

        self.spk_emb = self._prepare_emb(emb)

        self.target_rms = 0.1                  # muc RMS LibriMix luc training
        self.min_rms = SILENCE_RMS
        self.mask_power = float(mask_power)
        self.output_gain = 10.0 ** (float(output_gain_db) / 20.0)
        self.lock_db = float(lock_db)

        # thong ke
        self.n_chunks = 0
        self.total_ms = 0.0

    # ------------------------------------------------------------------
    def _check_io(self):
        names = [i.name for i in self.session.get_inputs()]
        expected = ["inp", "spk_emb", "mr", "mi", "mm"]

        missing = [n for n in expected if n not in names]
        if missing:
            raise RuntimeError(
                "Model khong dung interface V49.\n"
                f"  can  : {expected}\n"
                f"  thay : {names}\n"
                "Day khong phai file v49_*.onnx?")

        out_names = [o.name for o in self.session.get_outputs()]
        if "est_r" not in out_names or "est_i" not in out_names:
            raise RuntimeError(f"Output khong dung: {out_names}")

        # Model export khong co dynamic_axes -> T co dinh. Lay T tu graph.
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
              f"({'co dinh' if self.fixed_length else 'dong'})")
        print(f"  chunk       : {self.chunk_sec:.2f} s "
              f"({self.chunk_samples} samples), hop "
              f"{self.hop_samples / SR:.2f} s")

    # ------------------------------------------------------------------
    def _prepare_emb(self, emb):
        if isinstance(emb, str):
            if not os.path.exists(emb):
                raise FileNotFoundError(
                    f"Khong tim thay embedding: {emb}\n"
                    "Tao bang: python voice_lock.py enroll --seconds 10")
            emb = np.load(emb)

        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)

        if emb.shape[1] != SPK_DIM:
            raise ValueError(
                f"Embedding phai la {SPK_DIM}-d, dang co {emb.shape[1]}-d")

        # L2 normalize cho chac (WavLM output da normalize san)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    def process(self, chunk):
        """Chay 1 chunk. Tra ve (extracted, stats dict)."""
        chunk = np.asarray(chunk, dtype=np.float32)

        # Model co shape tinh -> pad / cat cho dung do dai
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

        # Chuan hoa muc vao cho khop domain training
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

        # Mask hieu dung MA MODEL TAO RA, truoc khi mu len power.
        # Day la chi so chan doan quan trong nhat:
        #   ~0.85  model tu tin day la giong dich
        #   ~0.50  model khong chac -> nghi ngo embedding
        # Lay trung binh co trong so theo bien do, vi cac bin gan nhu trong
        # co ty le est/mix vo nghia.
        est_mag = np.sqrt(est_r ** 2 + est_i ** 2 + EPS)
        mask = np.clip(est_mag / (mm_b + EPS), 0.0, 1.0)

        weight = float(mm_b.sum())
        mask_mean = float((mm_b * mask).sum() / (weight + 1e-12))

        # Mask sharpening: mu mask len power roi ap lai vao STFT goc
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
        # Bo phan output_gain khoi phep do de nguong khong bi lech
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
# MODE: enroll — tao speaker embedding (can torch + transformers)
# ============================================================================

def cmd_enroll(args):
    import soundfile as sf

    print("=" * 62)
    print("ENROLLMENT — tao speaker embedding 512-d (WavLM)")
    print("=" * 62)

    if args.wav:
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise SystemExit(
                f"File {args.wav} la {sr} Hz, can {SR} Hz. "
                "Convert truoc: sox in.wav -r 16000 -c 1 out.wav")
        print(f"  doc: {args.wav} ({len(audio) / SR:.1f} s)")
    else:
        import sounddevice as sd

        print(f"\n  Ghi am {args.seconds:.0f} s. Noi tu nhien, thay doi ngu dieu,")
        print("  doc nhieu cau khac nhau -> embedding phan biet tot hon.\n")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("  GO\n")

        audio = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1,
                       dtype="float32", device=args.in_device)
        sd.wait()
        audio = audio.squeeze()

        sf.write(args.out_wav, audio, SR)
        print(f"  luu: {args.out_wav}")

    r, peak = rms(audio), float(np.abs(audio).max())
    print(f"  RMS={r:.4f}  peak={peak:.4f}")
    if r < 0.005:
        print("  CANH BAO: qua nho — kiem tra mic / tang gain (alsamixer)")
    if peak > 0.95:
        print("  CANH BAO: clipping — lui ra xa mic")

    encoder = args.encoder or _autodetect_encoder()

    emb, n_seg, consistency = _wavlm_embedding(audio, encoder)

    np.save(args.emb, emb)

    print(f"\n  embedding: {emb.shape} -> {args.emb}")
    print(f"  {n_seg} segment, consistency {consistency:.3f} "
          f"(>0.85 tot, <0.7 nhieu)")

    if consistency < 0.7:
        print("  Nen ghi lai o noi yen tinh hon.")

    print("\n  Test ngay:")
    print(f"    python voice_lock.py live --model \"models/v49_int8 1.onnx\" "
          f"--emb {args.emb} --power 3 --gain 3")


# Cac thu muc se do khi duong dan tuong doi khong ton tai. Cho phep chay
# script tu bat ky dau ma van thay models/ o goc project.
_SEARCH = ("models", "../models", "../../models", ".", "..", "../..")


def _resolve(path):
    """Tim file theo `path`, neu khong co thi do ten file trong _SEARCH."""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path

    base = os.path.basename(path)
    for folder in _SEARCH:
        candidate = os.path.join(folder, base)
        if os.path.exists(candidate):
            return candidate

    return path


def _autodetect_encoder():
    """Tu tim WavLM encoder ONNX trong cac vi tri thong thuong."""
    for name in ("wavlm_sv_int8.onnx", "wavlm_sv_fp32.onnx"):
        path = _resolve(name)
        if os.path.exists(path):
            return path
    return None


def _segment(audio):
    """Cat thanh cac segment 3 s chong lan, bo doan im lang."""
    seg_len = 3 * SR
    seg_hop = int(1.5 * SR)

    segments = [audio[s:s + seg_len]
                for s in range(0, len(audio) - seg_len + 1, seg_hop)]
    segments = [s for s in segments if rms(s) > 0.005]

    if not segments:
        print("  Khong co segment nao du to — dung toan bo audio")
        if len(audio) < seg_len:
            reps = int(np.ceil(seg_len / max(len(audio), 1)))
            audio = np.tile(audio, reps)[:seg_len]
        segments = [audio[:seg_len]]

    return segments


def _norm_segment(seg):
    """Chuan hoa mean/var — khop voi Wav2Vec2FeatureExtractor cua WavLM."""
    x = np.ascontiguousarray(seg, dtype=np.float32)[None]
    return ((x - x.mean(-1, keepdims=True))
            / np.sqrt(x.var(-1, keepdims=True) + 1e-7)).astype(np.float32)


def _wavlm_embedding(audio, encoder_path=None):
    """10 s -> nhieu segment 3 s chong lan -> trung binh embedding.

    Uu tien encoder ONNX (khong can torch). Neu khong co thi fallback
    ve transformers + torch.
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
    """Duong chinh tren Pi 5 — chi can onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("Thieu onnxruntime: pip install onnxruntime")

    encoder_path = _resolve(encoder_path)

    if not os.path.exists(encoder_path):
        raise SystemExit(
            f"Khong tim thay encoder ONNX: {encoder_path}\n"
            "Tao tren laptop bang: python export_wavlm_onnx.py")

    print(f"\n  Encoder ONNX: {encoder_path} (khong can PyTorch)")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1

    sess = ort.InferenceSession(encoder_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])

    names = [i.name for i in sess.get_inputs()]
    if "input_values" not in names:
        raise SystemExit(
            f"Encoder khong dung interface. Can 'input_values', thay {names}")

    print(f"  {len(segments)} segment...", end="", flush=True)

    embs = []
    for seg in segments:
        e = sess.run(["embeddings"], {"input_values": _norm_segment(seg)})[0]
        e = e.ravel().astype(np.float32)
        embs.append(e / (np.linalg.norm(e) + 1e-8))
        print(".", end="", flush=True)

    print(" xong")

    if embs[0].shape[0] != SPK_DIM:
        raise SystemExit(
            f"Encoder tra ve {embs[0].shape[0]}-d, V49 can {SPK_DIM}-d")

    return embs


def _embed_torch(segments):
    """Fallback khi chua export encoder — can torch + transformers."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import WavLMForXVector
    except ImportError:
        raise SystemExit(
            "Khong co encoder ONNX va cung khong co PyTorch.\n"
            "Chon 1 trong 2:\n"
            "  a) Enroll tren laptop roi copy speaker_emb.npy sang Pi\n"
            "  b) Tren laptop chay export_wavlm_onnx.py, copy file .onnx sang\n"
            "     Pi, roi enroll voi --encoder wavlm_sv_int8.onnx")

    print("\n  Load WavLM-base-plus-sv (PyTorch)...")
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
# MODE: inspect — chan doan chat luong embedding
# ============================================================================

def cmd_inspect(args):
    print("=" * 62)
    print("PHAN TICH EMBEDDING")
    print("=" * 62)

    loaded = []

    for path in args.emb:
        resolved = _resolve(path)

        if not os.path.exists(resolved):
            print(f"\n  {path}: KHONG TIM THAY")
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
              f"{'' if abs(norm - 1.0) < 0.01 else '   <-- khong phai vector don vi'}")
        print(f"    mean/std   : {flat.mean():+.5f} / {flat.std():.5f}")
        print(f"    min/max    : {flat.min():+.4f} / {flat.max():+.4f}")
        print(f"    |x| < 1e-6 : {near_zero} / {flat.size}")
        print(f"    NaN / Inf  : {n_nan} / {n_inf}")

        # --- cac dang hong ---
        problems = []

        if flat.size != SPK_DIM:
            problems.append(
                f"kich thuoc {flat.size}, V49 can {SPK_DIM} -> khong dung duoc")
        if n_nan or n_inf:
            problems.append("co NaN/Inf -> file hong, enroll lai")
        if norm < 1e-6:
            problems.append(
                "toan so 0 -> luc enroll thu duoc im lang. Kiem tra mic "
                "(--in-device) va muc thu (alsamixer)")
        elif flat.std() < 1e-6:
            problems.append("moi phan tu gan nhu bang nhau -> embedding vo nghia")
        if near_zero > flat.size * 0.5:
            problems.append("qua nua phan tu bang 0 -> nghi ngo file hong")

        if problems:
            for p in problems:
                print(f"    -> LOI: {p}")
        else:
            print(f"    -> OK")
            loaded.append((label, flat / (norm + 1e-12)))

    # ----------------------------------------------------------------
    # So sanh tung cap
    # ----------------------------------------------------------------
    if len(loaded) > 1:
        print("\n  Cosine tung cap:")
        print()

        width = max(len(n) for n, _ in loaded) + 2
        print("  " + " " * width + "".join(f"{n[:9]:>10}" for n, _ in loaded))

        for name_a, a in loaded:
            row = f"  {name_a:<{width}}"
            for _, b in loaded:
                row += f"{float(np.dot(a, b)):>10.3f}"
            print(row)

    # ----------------------------------------------------------------
    # So voi ban ghi moi — phep kiem manh nhat
    # ----------------------------------------------------------------
    if args.wav:
        import soundfile as sf

        audio, sr = sf.read(_resolve(args.wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"{args.wav} la {sr} Hz, can {SR} Hz")

        print(f"\n  Tinh embedding tu {args.wav} ({len(audio)/SR:.1f} s)")

        encoder = args.encoder or _autodetect_encoder()
        fresh, n_seg, consistency = _wavlm_embedding(audio, encoder)

        print(f"  {n_seg} segment, consistency {consistency:.3f}")
        print("\n  Cosine voi cac embedding o tren:")

        for name, e in loaded:
            cos = float(np.dot(fresh, e))
            if cos > 0.85:
                verdict = "khop rat tot"
            elif cos > 0.7:
                verdict = "khop"
            elif cos > 0.5:
                verdict = "yeu — co the khac nguoi"
            else:
                verdict = "KHONG khop"
            print(f"    {name:<20} {cos:>6.3f}   {verdict}")

    # ----------------------------------------------------------------
    # Moc tham chieu
    # ----------------------------------------------------------------
    rng = np.random.default_rng(0)
    r = rng.normal(size=(2000, SPK_DIM))
    r /= np.linalg.norm(r, axis=1, keepdims=True)

    # abs PHAI dat ngoai tong: can |tich vo huong|, khong phai tong |a_i*b_i|
    cos_random = (r[:1000] * r[1000:]).sum(axis=1)
    baseline = float(np.abs(cos_random).mean())

    print("\n  Moc tham chieu:")
    print(f"    2 vector ngau nhien {SPK_DIM}-d : |cos| ~ {baseline:.3f}"
          f"  (do lech chuan {cos_random.std():.3f})")
    print("    cung nguoi, ban ghi khac  : thuong > 0.85")
    print("    khac nguoi                : thuong 0.5 - 0.8")
    print("\n  Cosine giua hai nguoi khac nhau ma > 0.95 nghia la embedding")
    print("  khong phan biet duoc ai — kiem tra lai buoc enroll.")


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

    print(f"\n  Benchmark ({n} lan, chunk {vl.chunk_sec:.1f} s, "
          f"{args.threads} threads)")
    print(f"    mean   : {times.mean():7.0f} ms")
    print(f"    median : {np.median(times):7.0f} ms")
    print(f"    min/max: {times.min():.0f} / {times.max():.0f} ms")
    print(f"    RTF    : {rtf:.3f}   "
          f"{'REALTIME OK' if rtf < 1.0 else 'QUA CHAM'}")

    budget = vl.hop_samples / SR * 1000.0
    print(f"\n    Ngan sach 1 hop = {budget:.0f} ms "
          f"-> {'du' if times.mean() < budget else 'THIEU, se drop audio'}")

    if rtf >= 1.0:
        print("\n  Cach tang toc tren Pi 5:")
        print("    - --power 1 (bo sharpening, ~20% nhanh hon)")
        print("    - --threads 4 va tat cac process nang khac")
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
        raise SystemExit(f"File la {sr} Hz, model can {SR} Hz.")

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
        # Ap MOT he so cho ca file, sau khi da do xong suppress -> khong
        # lam lech so do, va giu nguyen chenh lech giua LOCKED va REJECT.
        output = normalize_output(output, args.norm)
        print(f"\n  (--norm {args.norm}: peak {peak:.4f} -> "
              f"{float(np.abs(output).max()):.4f})")
    elif peak > 1.0:
        output /= peak
        print(f"\n  (chuan hoa lai vi clipping, peak was {peak:.2f})")

    sf.write(args.output, output, SR)

    _summary(vl, stats)
    print(f"\n  Da luu: {args.output}")


# ============================================================================
# MODE: verify — A/B giong dung vs giong sai
# ============================================================================

def cmd_verify(args):
    import soundfile as sf

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"File la {sr} Hz, model can {SR} Hz.")

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
        print(f"      suppression trung binh: {results[label.strip()]:+.1f} dB")

    gap = results["TARGET"] - results["OTHER"]

    print("\n" + "=" * 62)
    print("KET QUA VOICE LOCK")
    print("=" * 62)
    print(f"  giong dung (target) : {results['TARGET']:+6.1f} dB  (cang gan 0 cang tot)")
    print(f"  giong sai  (other)  : {results['OTHER']:+6.1f} dB  (cang am cang tot)")
    print(f"  khoang cach         : {gap:+6.1f} dB")

    if gap > 6.0:
        print("\n  => LOCK TOT: model phan biet ro theo enrollment.")
    elif gap > 3.0:
        print("\n  => LOCK YEU: nen enroll lai 10-15 s, tang --power.")
    else:
        print("\n  => KHONG LOCK: kiem tra lai embedding / file dau vao.")


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
                 "suppress_db": 0.0, "ms": 0.0},
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
          f"nguong lock {vl.lock_db:+.1f} dB")
    print("  DEO TAI NGHE — dung loa se bi hu (feedback).")
    if args.save:
        print(f"  ghi ra: {args.save}_input.wav / {args.save}_output.wav")
    print("  Ctrl+C de dung.\n")
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
            print("\n\n  Dung.")

    _summary(vl, state["hist"])

    if args.save and rec_in:
        import soundfile as sf
        a = np.concatenate(rec_in)
        b = np.concatenate(rec_out)
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]

        if getattr(args, "norm", "off") != "off":
            # Chi chuan hoa file da luu, khong anh huong am thanh phat ra
            # luc chay va cung khong anh huong so do suppress.
            b = normalize_output(b, args.norm)
            print(f"  (--norm {args.norm} ap cho file output)")

        sf.write(f"{args.save}_input.wav", a, SR)
        sf.write(f"{args.save}_output.wav", b, SR)
        print(f"\n  Da luu {L / SR:.1f} s:")
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
    print("TONG KET")
    print("=" * 62)

    if vl.n_chunks == 0 or not stats:
        print("  Khong co chunk nao co tieng.")
        return

    sup = np.array([s["suppress_db"] for s in stats])
    locked = sum(1 for s in stats if s["state"] == "LOCKED")

    print(f"  chunk co tieng : {len(stats)}")
    print(f"  LOCKED         : {locked} ({100 * locked / len(stats):.0f}%)")
    print(f"  REJECT         : {len(stats) - locked}")
    print(f"  suppress       : trung binh {sup.mean():+.1f} dB, "
          f"min {sup.min():+.1f} / max {sup.max():+.1f}")

    mk = np.array([s["mask"] for s in stats])
    print(f"  mask cua model : trung binh {mk.mean():.3f}  "
          f"({_mask_verdict(mk.mean())})")
    print(f"  latency        : {vl.total_ms / vl.n_chunks:.0f} ms/chunk, "
          f"RTF {vl.rtf:.3f} "
          f"({'realtime OK' if vl.rtf < 1.0 else 'QUA CHAM'})")


def cmd_devices(_args):
    import sounddevice as sd
    print(sd.query_devices())
    print(f"\n  default: {sd.default.device}")


# ============================================================================
# FILE CONFIG
#
# Dang key = value, '#' la ghi chu. Gia tri trong file duoc dung lam
# DEFAULT cua argparse, nen tham so dong lenh tu dong ghi de len no.
# Thu tu uu tien: mac dinh <  voice_lock.conf  <  dong lenh
# ============================================================================

CONFIG_FILE = "voice_lock.conf"

# key -> ham ep kieu
CONFIG_KEYS = {
    "model": str, "emb": str, "encoder": str,
    "power": float, "gain": float, "lock_db": float,
    "threads": int, "iters": int, "seconds": float,
    "in_device": str, "out_device": str,
    "save": str, "input": str, "output": str, "out_wav": str,
    "norm": str,
}


def _load_config(path=None):
    """Doc file config. Tra ve (dict gia tri, duong dan da dung)."""
    explicit = path is not None
    found = _resolve(path or CONFIG_FILE)

    if not os.path.exists(found):
        if explicit:
            raise SystemExit(f"Khong tim thay file config: {path}")
        return {}, None

    cfg = {}

    with open(found, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            if "=" not in line:
                raise SystemExit(
                    f"{found}:{lineno}: thieu dau '=' o dong: {line}")

            key, value = line.split("=", 1)
            key = key.strip().replace("-", "_")
            value = value.strip()

            if not value:
                continue                      # key bo trong = giu mac dinh

            if key not in CONFIG_KEYS:
                # Go sai ten key ma bo qua im lang la cai bay: nguoi dung
                # tuong da doi cau hinh nhung thuc te khong.
                valid = "\n".join(f"  {k}" for k in sorted(CONFIG_KEYS))
                raise SystemExit(
                    f"{found}:{lineno}: key khong hop le '{key}'\n"
                    f"  Key hop le:\n{valid}")

            try:
                cfg[key] = CONFIG_KEYS[key](value)
            except ValueError:
                raise SystemExit(
                    f"{found}:{lineno}: '{value}' khong doi duoc sang "
                    f"{CONFIG_KEYS[key].__name__} cho key '{key}'")

    return cfg, found


def _prescan_config(argv):
    """Tim --config / --no-config truoc khi argparse chay."""
    if "--no-config" in argv:
        return None, True

    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 >= len(argv):
            raise SystemExit("Thieu duong dan sau --config")
        return argv[i + 1], False

    return None, False


# ============================================================================
# CLI
# ============================================================================

def main():
    argv = [a for a in sys.argv[1:]]

    config_path, skip = _prescan_config(argv)
    cfg, cfg_used = ({}, None) if skip else _load_config(config_path)

    # Bo --config/--no-config ra khoi argv de argparse khong phai biet toi
    argv = [a for a in argv if a != "--no-config"]
    if config_path is not None:
        i = argv.index("--config")
        del argv[i:i + 2]

    def d(key, fallback):
        """Gia tri mac dinh: lay tu config neu co, khong thi dung fallback."""
        return cfg.get(key, fallback)

    p = argparse.ArgumentParser(
        description="V49 voice lock test — Raspberry Pi 5",
        epilog="Dat cac gia tri hay dung vao voice_lock.conf "
               "roi khoi go moi lan.\n"
               "Thu tu uu tien: mac dinh < voice_lock.conf < dong lenh.\n"
               "Bo qua config: --no-config    File khac: --config PATH",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_model_args(sp, need_emb=True):
        sp.add_argument("--model", default=d("model", "v49_int8.onnx"),
                        help="file .onnx (tu do trong models/)")
        if need_emb:
            sp.add_argument("--emb", default=d("emb", "speaker_emb.npy"),
                            help="speaker embedding 512-d (.npy)")
        sp.add_argument("--power", type=float, default=d("power", 2.0),
                        help="mask sharpening (1=tat, 2=mac dinh, 3=manh)")
        sp.add_argument("--gain", type=float, default=d("gain", 2.0),
                        help="gain dau ra (dB)")
        sp.add_argument("--lock-db", type=float,
                        default=d("lock_db", DEFAULT_LOCK_DB),
                        dest="lock_db", help="nguong quyet dinh LOCKED (dB)")
        sp.add_argument("--threads", type=int, default=d("threads", 4),
                        help="so thread ONNX (Pi 5 co 4 core)")
        sp.add_argument("--norm", choices=("off", "peak"),
                        default=d("norm", "off"),
                        help="chuan hoa am luong FILE dau ra "
                             "(peak = keo len 0.95). Khong lam lech so do "
                             "suppress vi ap sau khi do xong.")
        sp.add_argument("--config", metavar="PATH",
                        help="file config (mac dinh tu tim voice_lock.conf)")
        sp.add_argument("--no-config", action="store_true",
                        help="bo qua file config")

    # enroll
    sp = sub.add_parser("enroll", help="tao speaker embedding")
    sp.add_argument("--encoder", default=d("encoder", None),
                    help="WavLM encoder ONNX (khuyen dung tren Pi 5 — "
                         "khong can torch). Tao bang export_wavlm_onnx.py")
    sp.add_argument("--wav", default=None, help="dung file wav 16 kHz co san")
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
    sp = sub.add_parser("bench", help="do latency / RTF tren Pi 5")
    add_model_args(sp)
    sp.add_argument("--iters", type=int, default=d("iters", 20))
    sp.set_defaults(func=cmd_bench)

    # file
    sp = sub.add_parser("file", help="tach giong tu file wav")
    add_model_args(sp)
    sp.add_argument("-i", "--input", default=d("input", None),
                    required="input" not in cfg)
    sp.add_argument("-o", "--output", default=d("output", "extracted.wav"))
    sp.set_defaults(func=cmd_file)

    # verify
    sp = sub.add_parser("verify", help="A/B giong dung vs giong sai")
    add_model_args(sp, need_emb=False)
    sp.add_argument("--emb-target", required=True, dest="emb_target")
    sp.add_argument("--emb-other", required=True, dest="emb_other")
    sp.add_argument("-i", "--input", default=d("input", None),
                    required="input" not in cfg)
    sp.set_defaults(func=cmd_verify)

    # live
    sp = sub.add_parser("live", help="test realtime qua mic")
    add_model_args(sp)
    sp.add_argument("--save", default=d("save", None),
                    help="prefix luu wav in/out")
    sp.add_argument("--in-device", default=d("in_device", None),
                    dest="in_device")
    sp.add_argument("--out-device", default=d("out_device", None),
                    dest="out_device")
    sp.set_defaults(func=cmd_live)

    # inspect
    sp = sub.add_parser("inspect", help="chan doan chat luong embedding")
    sp.add_argument("--emb", nargs="+", default=[d("emb", "speaker_emb.npy")],
                    help="mot hoac nhieu file .npy de so sanh")
    sp.add_argument("--wav", default=None,
                    help="ban ghi moi de doi chieu voi embedding da luu")
    sp.add_argument("--encoder", default=d("encoder", None))
    sp.add_argument("--config", metavar="PATH")
    sp.add_argument("--no-config", action="store_true")
    sp.set_defaults(func=cmd_inspect)

    # devices
    sp = sub.add_parser("devices", help="liet ke thiet bi audio")
    sp.set_defaults(func=cmd_devices)

    args = p.parse_args(argv)
    args.config_used = cfg_used

    for name in ("in_device", "out_device"):
        v = getattr(args, name, None)
        if isinstance(v, str) and v.isdigit():
            setattr(args, name, int(v))

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nHuy.")
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nLOI: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
