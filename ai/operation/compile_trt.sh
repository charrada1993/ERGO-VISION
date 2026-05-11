#!/bin/bash
# ai/operation/compile_trt.sh
# Compiles ONNX model to TensorRT Engine with FP16 precision.

ONNX_MODEL="ai/models/ergo_net.onnx"
ENGINE_FILE="ai/models/ergo_net.engine"

echo "[TRT] Starting TensorRT compilation..."

# 1. Locate trtexec
TRT_EXEC=$(which trtexec)
if [ -z "$TRT_EXEC" ]; then
    # Common Jetson paths
    PATHS=("/usr/src/tensorrt/bin/trtexec" "/usr/bin/trtexec" "/usr/local/cuda/bin/trtexec")
    for p in "${PATHS[@]}"; do
        if [ -f "$p" ]; then
            TRT_EXEC=$p
            break
        fi
    done
fi

if [ -z "$TRT_EXEC" ]; then
    echo "[TRT] ERROR: trtexec not found. Please install TensorRT."
    exit 1
fi

if [ ! -f "$ONNX_MODEL" ]; then
    echo "[TRT] ERROR: ONNX model not found at $ONNX_MODEL. Run export_onnx.py first."
    exit 1
fi

echo "[TRT] Using: $TRT_EXEC"
echo "[TRT] Model: $ONNX_MODEL"
echo "[TRT] Output: $ENGINE_FILE"

# 2. Compile with FP16 and optimization flags
# --fp16: Use Half-precision (much faster on Jetson)
# --saveEngine: Path to save the serialized engine
# --verbose: Optional, but helpful for debugging
# --workspace=1024: Memory budget in MB
$TRT_EXEC \
    --onnx=$ONNX_MODEL \
    --saveEngine=$ENGINE_FILE \
    --fp16 \
    --workspace=1024 \
    --avgRuns=10 \
    --minTiming=1 \
    --useCudaGraph

if [ $? -eq 0 ]; then
    echo "[TRT] Compilation SUCCESS: $ENGINE_FILE"
else
    echo "[TRT] Compilation FAILED."
    exit 1
fi
