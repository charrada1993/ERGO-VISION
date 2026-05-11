# ONNX: Open Neural Network Exchange in ERGO-VISION

## What is ONNX?
**ONNX (Open Neural Network Exchange)** is an open-source format designed to represent machine learning models. Think of it as a "universal translator" for AI. It allows you to train a model in one framework (like PyTorch or TensorFlow) and run it in another (like TensorRT or ONNX Runtime).

### Why use ONNX for ERGO-VISION?
1.  **Framework Independence**: You can develop `ErgoNet` in PyTorch on your laptop, but run it on the Jetson without needing the full PyTorch library (which is heavy and slow on ARM).
2.  **Optimized Inference**: ONNX is the required "bridge" to reach **TensorRT**. TensorRT cannot read PyTorch files (`.pt` or `.pth`) directly; it needs the standardized graph structure of an ONNX file.
3.  **Hardware Acceleration**: ONNX allows the Jetson to understand the model's math in a way that can be mapped directly to the **CUDA cores** and **DLA (Deep Learning Accelerator)**.

---

## How to implement ONNX in this Project

### 1. Exporting the Model
We use the `torch.onnx.export` function to "freeze" the model weights and the computation graph into a `.onnx` file.

**Key Parameters we use:**
- `input_names`: Labels the input as `"landmarks"`.
- `output_names`: Labels our multi-head outputs (`"angles"`, `"scores"`, `"anomalies"`).
- `opset_version`: We use **Version 17**, which supports the advanced operations used in `ErgoNet`.

### 2. The Conversion Pipeline
In this project, the workflow is:
1.  **PyTorch (`.py`)** → Define the architecture.
2.  **ONNX (`.onnx`)** → Export the "frozen" graph.
3.  **TensorRT (`.engine`)** → Compile for the Jetson Orin GPU.

---

## Commands to Generate ONNX
Once your environment is set up, you can generate the ONNX file using the operation script:

```bash
python3 ai/operation/export_onnx.py
```

This will produce `ai/models/ergo_net.onnx`, which is ready for TensorRT compilation.
