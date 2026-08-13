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

ONNX Runtime C++ (aarch64) is also required. By default both the build script and CMake look for it at `<repo>/onnxruntime-linux-aarch64-1.29.0`, which is already unpacked in this repo. Pi OS must be 64-bit.

## Build

```bash
cd cpp_test
./build.sh              # build Release -> build/v49_pi5
```

Options:

```bash
./build.sh --deps       # apt install build deps, then build
./build.sh --clean      # wipe build/ and reconfigure
./build.sh --debug      # Debug build
./build.sh --fetch-ort  # download onnxruntime-linux-aarch64-1.29.0 if missing
./build.sh --ort /path/to/onnxruntime-linux-aarch64-1.29.0
./build.sh -j 4         # limit parallel jobs
```

Manual equivalent:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$PWD/../onnxruntime-linux-aarch64-1.29.0
cmake --build build -j"$(nproc)"
```

The ONNX Runtime `lib/` directory is baked into the binary's RPATH, so no `LD_LIBRARY_PATH` is needed at runtime as long as that directory stays in place.

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

## Audio devices

The Pi 5 has no built-in microphone, so a USB mic or sound card is required for
real-time mode. List what PortAudio sees:

```bash
./build/v49_pi5 --list-devices
```

Then select by index or by name substring:

```bash
./build/v49_pi5 --model v49_int8.onnx --emb speaker_emb.npy \
  --in-device "USB" --out-device 1
```

PortAudio probes every ALSA/JACK/OSS backend at startup and each unavailable one
prints to stderr (`jack server is not running`, `Cannot open device /dev/dsp`,
`Unknown PCM iec958...`). These are harmless and are suppressed by default; pass
`--verbose-audio` to see them.

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
