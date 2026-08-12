#!/usr/bin/env python3

import wave
import pyaudio


# =========================
# Configuration
# =========================

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

OUTPUT_FILE = "record.wav"


# =========================
# Recording
# =========================

audio = pyaudio.PyAudio()

print("Available input devices:")
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)

    if info["maxInputChannels"] > 0:
        print(
            f"[{i}] {info['name']}"
        )

print()
print("Press ENTER to start recording...")
input()

print("Recording...")
print("Press Ctrl+C to stop.")

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

frames = []

try:
    while True:
        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        frames.append(data)

except KeyboardInterrupt:
    print("\nStopping recording...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()


# =========================
# Save WAV
# =========================

with wave.open(OUTPUT_FILE, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(
        audio.get_sample_size(FORMAT)
    )
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(
        b"".join(frames)
    )


print()
print("Recording saved:")
print(f"  {OUTPUT_FILE}")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Channels:    {CHANNELS}")

