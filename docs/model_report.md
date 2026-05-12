# ErgoNet v2.0: Technical Model Report 🧠

*Last updated: 2026-05-11 — ErgoNet v2.0 training complete.*

---

## 1. Model Type & Architecture

**ErgoNet v2.0** is a **Multi-Output Multi-Layer Perceptron (MLP)**. Unlike traditional classification models that predict a single label, this architecture uses a shared feature-extraction layer to simultaneously predict four distinct ergonomic outcomes from a single forward pass.

### Specifications

| Component | Details |
|---|---|
| **Input Layer** | 12 nodes — normalized biomechanical joint angles |
| **Hidden Layer** | 512 nodes — ReLU activation |
| **Output Layer** | 4 nodes — multi-task regression heads |
| **Initialization** | Xavier (Glorot) — prevents gradient vanishing/explosion |
| **Optimizer** | Gradient Descent, lr = 0.005, 500 epochs |
| **Framework** | Pure NumPy — zero external ML dependencies |

### Input Features (12 Joint Angles)

| Feature | Description |
|---|---|
| `Neck_Flexion_deg` | Neck forward/backward tilt |
| `Trunk_Flexion_deg` | Trunk forward lean |
| `R/L_Shoulder_Flexion_deg` | Shoulder elevation (bilateral) |
| `R/L_Elbow_Flexion_deg` | Elbow angle (bilateral) |
| `R/L_Wrist_Deviation_deg` | Wrist radial/ulnar deviation (bilateral) |
| `R/L_Hip_Flexion_deg` | Hip angle (bilateral) |
| `R/L_Knee_Flexion_deg` | Knee angle (bilateral) |

### Output Heads (Multi-Task)

| Head | Type | Range |
|---|---|---|
| **Risk Score** | Continuous regression | 0.0 – 10.0 |
| **Severity Code** | Categorical | 0 (Healthy) → 4 (Critical) |
| **Location Code** | Anatomical segment ID | 0 – N |
| **Condition Code** | Diagnostic prediction | e.g. Tendinitis, Strain |

---

## 2. Rationale: Why this Architecture?

### A. Dependency-Free Deployment (NumPy)
The model is implemented in **pure NumPy**. This is critical for the **NVIDIA Jetson Orin** environment because:
- Eliminates the need for PyTorch/TensorFlow/Keras (~2 GB RAM at load-time).
- Bypasses complex dependency conflicts on the Jetson's ARM64 architecture.
- The `.pkl` model file is lightweight (<5 MB) and loads in milliseconds.

### B. High-Speed Inference
Vectorized matrix multiplications in NumPy using OpenBLAS on ARM v8.2 NEON yield **~8 ms** inference latency. The AI can process full 30 FPS camera feeds without becoming a pipeline bottleneck.

### C. Angle-Based Invariance (v2.0 Improvement)
ErgoNet v1.0 used raw (x, y, z) MediaPipe landmarks. v2.0 switches to **computed joint angles**.
- **Why?** Raw landmarks change based on the subject's distance from the camera. Joint angles are **mathematically invariant** to camera distance and perspective distortion, resulting in significantly more stable predictions in real-world industrial environments.

---

## 3. Forward Pass Logic

```python
# Layer 1
z1 = X @ W1 + b1      # (N, 12) × (12, 512) → (N, 512)
a1 = ReLU(z1)          # Non-linear activation

# Layer 2 (Output)
output = a1 @ W2 + b2  # (N, 512) × (512, 4) → (N, 4)
```

The output vector is then de-normalized using the stored `y_mean` and `y_std` and sliced into the four diagnostic heads, which are served to the Dashboard via Socket.IO.

---

## 4. Training Pipeline (`ai/train_v2.py`)

```bash
cd ~/ERGO-VISION/ai
python3 train_v2.py
```

The script:
1. Loads `ai/data/dataset_TMS_enriched.csv` (~20,000 samples).
2. Z-score normalizes inputs (X) and outputs (y).
3. Runs a custom 500-epoch gradient descent loop, printing loss + accuracy every epoch.
4. Saves the frozen model to `ai/models/ergo_net_v2.pkl`.
5. Writes the epoch-by-epoch training log to `ai/data/training_log.json`.
6. Displays training/validation accuracy & loss curves via matplotlib.

---

## 5. Performance Metrics (v2.0 — Completed Run)

| Metric | Value |
|---|---|
| **Training Dataset** | 20,000+ TMS enriched samples (`dataset_TMS_enriched.csv`) |
| **Epochs** | 500 |
| **Final Training Loss (MSE)** | **0.2742** |
| **Final Training Accuracy** | **97.14%** |
| **Final Validation Loss** | **0.2971** |
| **Final Validation Accuracy** | **94.22%** |
| **Inference Latency** | ~8 ms (Jetson Orin CPU) |
| **Model File** | `ai/models/ergo_net_v2.pkl` |

---

## 6. Saved Model Structure

```python
state = {
    'version': '2.0',
    'W1': np.ndarray,   # (12, 512)
    'b1': np.ndarray,   # (1, 512)
    'W2': np.ndarray,   # (512, 4)
    'b2': np.ndarray,   # (1, 4)
    'X_mean': np.ndarray,   # per-feature input mean
    'X_std':  np.ndarray,   # per-feature input std
    'y_mean': np.ndarray,   # per-target output mean
    'y_std':  np.ndarray,   # per-target output std
    'input_cols':  list,    # 12 angle column names
    'target_cols': list     # ['risk_score', 'severity_code', 'location_code', 'condition_code']
}
```

---

*Documented by ErgoVision AI Team · 2026*
