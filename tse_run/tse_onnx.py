#!/usr/bin/env python3

import argparse
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort


# ============================================================
# Configuration
# ============================================================

TARGET_SR = 16000


# ============================================================
# ONNX helper
# ============================================================

def create_session(model_path):
    print(f"\nLoading model: {model_path}")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    print("Inputs:")
    for x in session.get_inputs():
        print(
            f"  name={x.name}, "
            f"shape={x.shape}, "
            f"type={x.type}"
        )

    print("Outputs:")
    for x in session.get_outputs():
        print(
            f"  name={x.name}, "
            f"shape={x.shape}, "
            f"type={x.type}"
        )

    return session


# ============================================================
# Audio
# ============================================================

def load_audio(path, target_sr=TARGET_SR):
    print(f"Loading audio: {path}")

    audio, sr = sf.read(path, dtype="float32")

    # Stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample
    if sr != target_sr:
        print(f"Resampling {sr} Hz -> {target_sr} Hz")

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sr
        )

    audio = audio.astype(np.float32)

    # Normalize only if necessary
    peak = np.max(np.abs(audio))

    if peak > 1.0:
        audio = audio / peak

    print(
        f"  samples : {len(audio)}"
    )
    print(
        f"  duration: {len(audio) / target_sr:.2f} sec"
    )

    return audio


# ============================================================
# ECAPA speaker encoder
# ============================================================

def extract_embedding(session, audio):
    """
    Run speaker encoder ONNX.

    This assumes the encoder has one audio input.
    """

    inputs = session.get_inputs()

    if len(inputs) != 1:
        raise RuntimeError(
            "Speaker encoder must have exactly one input. "
            f"Found {len(inputs)} inputs."
        )

    input_info = inputs[0]
    input_name = input_info.name

    # Most audio ONNX models expect:
    # [batch, samples]
    x = audio.astype(np.float32)

    x = np.expand_dims(x, axis=0)

    print("\nRunning speaker encoder...")

    outputs = session.run(
        None,
        {
            input_name: x
        }
    )

    embedding = outputs[0]

    print(
        "Raw embedding shape:",
        embedding.shape
    )

    embedding = embedding.astype(np.float32)

    # Flatten if necessary
    embedding = embedding.reshape(embedding.shape[0], -1)

    # L2 normalization
    norm = np.linalg.norm(
        embedding,
        axis=1,
        keepdims=True
    )

    embedding = embedding / (norm + 1e-8)

    print(
        "Normalized embedding shape:",
        embedding.shape
    )

    return embedding


# ============================================================
# TSE inference
# ============================================================

def run_tse(session, mixture, embedding):

    inputs = session.get_inputs()

    print("\nTSE model inputs:")

    for i, inp in enumerate(inputs):
        print(
            f"[{i}] "
            f"name={inp.name}, "
            f"shape={inp.shape}, "
            f"type={inp.type}"
        )

    if len(inputs) != 2:
        raise RuntimeError(
            "This script expects the TSE model "
            "to have exactly 2 inputs:\n"
            "  1. mixture\n"
            "  2. speaker embedding\n"
            f"Found {len(inputs)} inputs."
        )

    # --------------------------------------------------------
    # Detect audio input and embedding input
    # --------------------------------------------------------

    audio_input = None
    embedding_input = None

    for inp in inputs:

        shape = inp.shape

        shape_str = str(shape).lower()

        # Heuristic:
        # embedding usually has a relatively small dimension
        # such as [1, 192], [1, 256], [1, 512]
        if len(shape) == 2:

            last_dim = shape[-1]

            if isinstance(last_dim, int):
                if last_dim <= 1024:
                    embedding_input = inp
                    continue

        # Otherwise assume audio
        audio_input = inp

    # Fallback
    if audio_input is None or embedding_input is None:

        print(
            "\nCould not automatically detect inputs."
        )

        audio_input = inputs[0]
        embedding_input = inputs[1]

    print(
        "\nSelected audio input:",
        audio_input.name
    )

    print(
        "Selected embedding input:",
        embedding_input.name
    )

    # --------------------------------------------------------
    # Prepare audio
    # --------------------------------------------------------

    x_audio = mixture.astype(np.float32)

    audio_shape = audio_input.shape

    if len(audio_shape) == 2:

        # [B, T]
        x_audio = np.expand_dims(
            x_audio,
            axis=0
        )

    elif len(audio_shape) == 3:

        # Usually [B, C, T]
        x_audio = np.expand_dims(
            x_audio,
            axis=0
        )

    else:
        raise RuntimeError(
            f"Unsupported audio input shape: {audio_shape}"
        )

    # --------------------------------------------------------
    # Prepare embedding
    # --------------------------------------------------------

    x_embedding = embedding.astype(np.float32)

    expected_shape = embedding_input.shape

    print(
        "Embedding expected shape:",
        expected_shape
    )

    print(
        "Embedding actual shape:",
        x_embedding.shape
    )

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    feed = {
        audio_input.name: x_audio,
        embedding_input.name: x_embedding
    }

    print("\nRunning TSE model...")

    outputs = session.run(
        None,
        feed
    )

    print(
        "Number of outputs:",
        len(outputs)
    )

    for i, out in enumerate(outputs):
        print(
            f"Output[{i}] shape:",
            out.shape
        )

    result = outputs[0]

    # --------------------------------------------------------
    # Convert output to 1D waveform
    # --------------------------------------------------------

    result = np.asarray(result)

    print(
        "Raw output shape:",
        result.shape
    )

    # Remove batch/channel dimensions
    result = np.squeeze(result)

    if result.ndim != 1:
        raise RuntimeError(
            f"Cannot convert model output "
            f"to waveform. Shape={result.shape}"
        )

    result = result.astype(np.float32)

    # Prevent clipping
    peak = np.max(np.abs(result))

    if peak > 1.0:
        result = result / peak

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="ONNX Runtime Target Speaker Extraction"
    )

    parser.add_argument(
        "--enroll",
        required=True,
        help="Enrollment speaker WAV"
    )

    parser.add_argument(
        "--mixed",
        required=True,
        help="Mixed audio WAV"
    )

    parser.add_argument(
        "--encoder",
        required=True,
        help="ECAPA-TDNN ONNX model"
    )

    parser.add_argument(
        "--tse",
        required=True,
        help="TSE ONNX model"
    )

    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ONNX TARGET SPEAKER EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    encoder_session = create_session(
        args.encoder
    )

    tse_session = create_session(
        args.tse
    )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    enroll = load_audio(
        args.enroll
    )

    mixed = load_audio(
        args.mixed
    )

    # --------------------------------------------------------
    # Speaker embedding
    # --------------------------------------------------------

    embedding = extract_embedding(
        encoder_session,
        enroll
    )

    # --------------------------------------------------------
    # TSE
    # --------------------------------------------------------

    output = run_tse(
        tse_session,
        mixed,
        embedding
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    sf.write(
        args.output,
        output,
        TARGET_SR
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Output: {args.output}"
    )

    print(
        f"Duration: {len(output) / TARGET_SR:.2f} sec"
    )


if __name__ == "__main__":
    main()
    