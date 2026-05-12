# AI Operation: Inference-Only Engine

*Last updated: 2026-05-11*

This folder contains the **Operational AI** engine — the version of ErgoNet v2.0 used in the live dashboard. Unlike the training scripts, this code cannot learn or change. It is designed for **reliability, speed, and zero-dependency stability**.

---

## Model: ErgoNet v2.0

| Property | Value |
|---|---|
| **Model File** | `ai/models/ergo_net_v2.pkl` |
| **Architecture** | MLP: 12 → 512 (ReLU) → 4 |
| **Training Accuracy** | 97.14% |
| **Validation Accuracy** | 94.22% |
| **Inference Latency** | ~8 ms (Jetson Orin CPU) |
| **Dependencies** | `numpy` only |

---

## Key Characteristics

| Property | Detail |
|---|---|
| **Zero Overhead** | No training logic, no gradient computation, no backpropagation |
| **Deterministic** | Same posture input always produces the same diagnostic output |
| **Dependency-Lite** | Only `numpy` required — immune to library version conflicts |
| **Hardware-Agnostic** | Optimized for Jetson ARM64, but runs on any Python 3.10+ CPU |

---

## How It Works

1. At server startup, `inference.py` loads `ergo_net_v2.pkl` (weights + normalization stats).
2. For each camera frame, `pose/skeleton.py` computes a 12-element joint angle vector.
3. The angle vector is Z-score normalized using stored `X_mean` / `X_std`.
4. A single 2-layer forward pass executes:
   ```python
   a1 = np.maximum(0, X_norm @ W1 + b1)   # ReLU hidden layer
   output = a1 @ W2 + b2                    # Linear output
   ```
5. Output is de-normalized using stored `y_mean` / `y_std`.
6. The 4 output heads are extracted and emitted via Socket.IO to the dashboard.

---

## Input / Output Schema

**Input**: 12 normalized joint angles (float32 vector)
```
[Neck_Flexion, Trunk_Flexion, R_Shoulder, L_Shoulder,
 R_Elbow, L_Elbow, R_Wrist, L_Wrist,
 R_Hip, L_Hip, R_Knee, L_Knee]
```

**Output**: 4 diagnostic values
```
[risk_score (0–10), severity_code (0–4), location_code, condition_code]
```

---

## Files in This Folder

| File | Purpose |
|---|---|
| `inference.py` | Production inference engine (loads `.pkl`, runs forward pass) |
| `export_onnx.py` | Optional: exports model to ONNX for future TensorRT use |

---

*Documented by ErgoVision AI Team · 2026*
