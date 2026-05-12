# ONNX Export Guide for ErgoNet v2.0

*Last updated: 2026-05-11*

> **Note:** ErgoNet v2.0 is currently deployed as a **Pure NumPy `.pkl` model** for zero-dependency inference on the Jetson Orin. This guide documents the ONNX pathway for future TensorRT acceleration.

---

## What is ONNX?

**ONNX (Open Neural Network Exchange)** is a standardized, open-source format for representing ML models — a "universal translator" for AI. It allows a model trained in one framework (PyTorch, TensorFlow) to be executed in another (ONNX Runtime, TensorRT).

---

## Why ONNX for ERGO-VISION?

| Reason | Details |
|---|---|
| **Framework Independence** | Train in PyTorch on a workstation, deploy on Jetson without PyTorch |
| **TensorRT Bridge** | TensorRT cannot read `.pt` or `.pth` files directly — ONNX is the required intermediate format |
| **Hardware Acceleration** | ONNX → TensorRT maps model operations directly onto Jetson CUDA cores and DLA |
| **Portability** | `.onnx` files run on any hardware supported by ONNX Runtime |

---

## Current Model: NumPy `.pkl` (Active)

ErgoNet v2.0 currently runs as a frozen NumPy model:

```
ai/models/ergo_net_v2.pkl
```

| Property | Value |
|---|---|
| Format | Python pickle (`.pkl`) |
| Inference | Pure NumPy matrix multiply |
| Latency | ~8 ms (Jetson CPU) |
| Dependencies | `numpy` only |

---

## ONNX Export Pathway (Future)

To export ErgoNet v2.0 to ONNX for TensorRT acceleration:

### Step 1: Reconstruct in PyTorch
```python
import torch
import torch.nn as nn

class ErgoNetTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

### Step 2: Load Weights from `.pkl`
```python
import pickle, numpy as np, torch

with open('ai/models/ergo_net_v2.pkl', 'rb') as f:
    state = pickle.load(f)

model = ErgoNetTorch()
model.fc1.weight.data = torch.tensor(state['W1'].T, dtype=torch.float32)
model.fc1.bias.data   = torch.tensor(state['b1'].squeeze(), dtype=torch.float32)
model.fc2.weight.data = torch.tensor(state['W2'].T, dtype=torch.float32)
model.fc2.bias.data   = torch.tensor(state['b2'].squeeze(), dtype=torch.float32)
```

### Step 3: Export to ONNX
```python
dummy_input = torch.randn(1, 12)
torch.onnx.export(
    model, dummy_input,
    'ai/models/ergo_net_v2.onnx',
    input_names=['joint_angles'],
    output_names=['risk_score', 'severity_code', 'location_code', 'condition_code'],
    opset_version=17
)
```

### Step 4: Compile with TensorRT
```bash
trtexec --onnx=ai/models/ergo_net_v2.onnx \
        --saveEngine=ai/models/ergo_net_v2.engine \
        --fp16
```

### Step 5: Run via Operation Script
```bash
python3 ai/operation/export_onnx.py
```

This will produce `ai/models/ergo_net_v2.onnx`, ready for TensorRT compilation on the Jetson.

---

## ONNX Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `input_names` | `['joint_angles']` | 12-element normalized angle vector |
| `output_names` | 4 diagnostic heads | risk, severity, location, condition |
| `opset_version` | 17 | Supports all operations in ErgoNet v2.0 |

---

*Documented by ErgoVision AI Team · 2026*
