# V49 C++ / Raspberry Pi 5

This is a C++ port of `v49_pi5.py`.

## Runtime files

Copy these to the Pi:

- `v49_pi5`             executable
- `v49_int8.onnx`
- `speaker_emb.npy`

The C++ program keeps the model-side pipeline from the Python script:
- 16 kHz audio
- 512-point STFT
- 128 sample STFT hop
- 3 second processing chunks
- 1.5 second processing hop
- compressed-magnitude normalization, COMP=0.3
- ONNX inputs: inp, spk_emb, mr, mi, mm
- ONNX outputs: est_r, est_i
- frequency-dependent mask sharpening
- inverse STFT + overlap/add

## Pi packages

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
    libfftw3-dev libsndfile1-dev portaudio19-dev
```

ONNX Runtime C++ must also be installed on the Pi. The CMake file expects it under `/opt/onnxruntime` by default.

## Build

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build -j4
```

## File test

```bash
./build/v49_pi5 \
  --model v49_int8.onnx \
  --emb speaker_emb.npy \
  --file input.wav \
  -o output.wav
```

## Benchmark

```bash
./build/v49_pi5 \
  --model v49_int8.onnx \
  --emb speaker_emb.npy \
  --bench
```

## Real-time

```bash
./build/v49_pi5 \
  --model v49_int8.onnx \
  --emb speaker_emb.npy
```

With recording:

```bash
./build/v49_pi5 \
  --model v49_int8.onnx \
  --emb speaker_emb.npy \
  --save demo1
```

## Important

The original Python script uses NumPy's exact `reflect` padding and NumPy FFT/STFT behavior. The C++ version reproduces the intended algorithm with FFTW3, but before using it for production audio quality, compare:

`input.wav -> Python output.wav`
versus
`input.wav -> C++ output.wav`

using the same model and embedding.

Also note that the Python source reports a target Pi 5 RTF of roughly 0.2-0.4, but that is an expectation, not a measured result for this C++ port.
