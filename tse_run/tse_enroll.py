#!/usr/bin/env python3

import argparse
import time
import wave
import numpy as np
import pyaudio


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024


# ============================================================
# Record audio
# ============================================================

def record_audio(duration, output_file):

    audio = pyaudio.PyAudio()

    print()
    print("=" * 60)
    print("VOICE ENROLLMENT")
    print("=" * 60)

    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Channels    : {CHANNELS}")
    print(f"Duration    : {duration:.1f} sec")
    print()

    # --------------------------------------------------------
    # List input devices
    # --------------------------------------------------------

    print("Available input devices:")

    for i in range(audio.get_device_count()):

        info = audio.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:

            print(
                f"  [{i}] "
                f"{info['name']} "
                f"(inputs={info['maxInputChannels']})"
            )

    print()

    # --------------------------------------------------------
    # Countdown
    # --------------------------------------------------------

    print("Get ready...")

    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print()
    print(">>> RECORDING <<<")
    print("Speak normally...")
    print()

    # --------------------------------------------------------
    # Open microphone
    # --------------------------------------------------------

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []

    num_chunks = int(
        SAMPLE_RATE / CHUNK * duration
    )

    try:

        for _ in range(num_chunks):

            data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            frames.append(data)

    except KeyboardInterrupt:

        print("\nRecording interrupted.")

    finally:

        stream.stop_stream()
        stream.close()

        audio.terminate()

    print()
    print("Recording finished.")

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    raw_audio = b"".join(frames)

    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16
    )

    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    samples_float = samples.astype(
        np.float32
    )

    samples_float -= np.mean(
        samples_float
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    peak = np.max(
        np.abs(samples_float)
    )

    if peak > 0:

        # Keep some headroom
        samples_float *= (
            0.95 * 32767.0 / peak
        )

    samples = samples_float.astype(
        np.int16
    )

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    with wave.open(
        output_file,
        "wb"
    ) as wf:

        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        wf.writeframes(
            samples.tobytes()
        )

    print()
    print("=" * 60)
    print("ENROLLMENT CREATED")
    print("=" * 60)

    print(f"File       : {output_file}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels   : {CHANNELS}")
    print(
        f"Duration   : "
        f"{len(samples) / SAMPLE_RATE:.2f} sec"
    )

    print()
    print("Ready for ECAPA speaker embedding.")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Record speaker enrollment audio"
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="enroll.wav",
        help="Output WAV file"
    )

    args = parser.parse_args()

    record_audio(
        args.duration,
        args.output
    )


if __name__ == "__main__":
    main()

