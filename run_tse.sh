bash
#!/bin/bash

# ============================================================
# TSE ONNX Runner - Raspberry Pi 5
# ============================================================

# Activate virtual environment
source ~/tse_env/bin/activate

# Check input files
if [ ! -f "enroll.wav" ]; then
    echo "ERROR: enroll.wav not found"
    exit 1
fi

if [ ! -f "mixed.wav" ]; then
    echo "ERROR: mixed.wav not found"
    exit 1
fi

# Check models
if [ ! -f "models/ecapa.onnx" ]; then
    echo "ERROR: models/ecapa.onnx not found"
    exit 1
fi

if [ ! -f "models/v49_int8.onnx" ]; then
    echo "ERROR: models/v49_int8.onnx not found"
    exit 1
fi

echo "=========================================="
echo "        TSE ONNX INFERENCE"
echo "=========================================="

echo "Enroll : enroll.wav"
echo "Mixed  : mixed.wav"
echo "Encoder: models/ecapa.onnx"
echo "TSE    : models/v49_int8.onnx"
echo "Output : output.wav"
echo "=========================================="

# Run TSE
python tse_onnx.py \
    --enroll enroll.wav \
    --mixed mixed.wav \
    --encoder models/ecapa.onnx \
    --tse models/v49_int8.onnx \
    --output output.wav

# Check result
if [ $? -eq 0 ]; then
    echo
    echo "=========================================="
    echo "TSE completed successfully!"
    echo "Output: output.wav"
    echo "=========================================="
else
    echo
    echo "=========================================="
    echo "ERROR: TSE inference failed!"
    echo "=========================================="
    exit 1
fi

