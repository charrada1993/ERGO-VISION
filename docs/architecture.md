# AI Architecture: ErgoNet v2.0 Deep Dive

*Last updated: 2026-05-11*

**ErgoNet v2.0** is the active inference engine powering the ERGO-VISION real-time ergonomic pipeline. It is a custom-built Neural Network implemented in **Pure NumPy**, designed to operate on the **NVIDIA Jetson Orin Nano (reComputer J3011)** without requiring PyTorch, TensorFlow, or Keras.

---

## 1. Design Philosophy

Unlike standard pose-estimation models that output joint coordinates, ErgoNet v2.0 is a **Multi-Task Regression MLP**. A single forward pass simultaneously produces four clinical diagnostic outputs. This is more efficient than running four separate models and captures the cross-joint ergonomic relationships in a shared representation.

### Why Pure NumPy?
- **No Library Bloat**: Avoids the ~2 GB RAM overhead of PyTorch/TensorFlow at startup.
- **ARM64 Optimized**: Leverages Jetson's **OpenBLAS with ARM v8.2 NEON** instructions for vectorized matrix multiplication.
- **Inference Latency**: Single forward pass completes in **< 8 ms** on Jetson CPU.
- **Portable**: The frozen `.pkl` model file works on any Python 3.10+ environment — no CUDA required.

---

## 2. Network Topology

```
Input (12)  →  Hidden (512, ReLU)  →  Output (4)
```

| Layer | Shape | Activation |
|---|---|---|
| Input | `(N, 12)` | — (Z-score normalized angles) |
| Hidden | `(12 → 512)` | ReLU |
| Output | `(512 → 4)` | Linear (regression) |

### Weight Initialization
Xavier (Glorot) initialization is used to maintain variance across layers:
```python
W1 = np.random.randn(12, 512) * np.sqrt(1.0 / 12)
W2 = np.random.randn(512, 4)  * np.sqrt(1.0 / 512)
```
This prevents gradient vanishing/explosion and ensures the model converges cleanly on the Jetson.

---

## 3. Input: 12 Biomechanical Joint Angles

ErgoNet v2.0 moved from raw MediaPipe landmark coordinates (x, y, z) to **computed joint angles**. This is the key improvement over v1.0:

| Input Feature | Joint |
|---|---|
| `Neck_Flexion_deg` | Cervical spine forward/backward tilt |
| `Trunk_Flexion_deg` | Lumbar spine forward lean |
| `R_Shoulder_Flexion_deg` | Right shoulder elevation |
| `L_Shoulder_Flexion_deg` | Left shoulder elevation |
| `R_Elbow_Flexion_deg` | Right elbow angle |
| `L_Elbow_Flexion_deg` | Left elbow angle |
| `R_Wrist_Deviation_deg` | Right wrist radial/ulnar deviation |
| `L_Wrist_Deviation_deg` | Left wrist radial/ulnar deviation |
| `R_Hip_Flexion_deg` | Right hip flexion |
| `L_Hip_Flexion_deg` | Left hip flexion |
| `R_Knee_Flexion_deg` | Right knee angle |
| `L_Knee_Flexion_deg` | Left knee angle |

**Why angles instead of raw landmarks?**
Raw landmark coordinates (x, y, z) change based on camera distance and perspective. A person standing 1 m away produces entirely different x/y values than the same person at 3 m. Angles are **scale-invariant and perspective-invariant**, making ErgoNet v2.0 significantly more robust in real-world industrial deployments.

---

## 4. Output: 4 Diagnostic Heads

| Head | Type | Description |
|---|---|---|
| `risk_score` | Continuous (0.0–10.0) | Overall ergonomic risk magnitude |
| `severity_code` | Categorical int | 0=Healthy, 1=Low, 2=Moderate, 3=High, 4=Critical |
| `location_code` | Categorical int | Anatomical segment with highest risk |
| `condition_code` | Categorical int | Predicted TMS condition (Tendinitis, Strain, etc.) |

---

## 5. Forward Pass

```python
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1   # Linear transform
    self.a1 = np.maximum(0, self.z1)           # ReLU
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    return self.z2
```

---

## 6. Training Summary

| Parameter | Value |
|---|---|
| Dataset | `dataset_TMS_enriched.csv` (~20,000 samples) |
| Epochs | 500 |
| Learning Rate | 0.005 |
| Loss Function | Mean Squared Error (MSE) |
| Final Training Loss | **0.2742** |
| Final Training Accuracy | **97.14%** |
| Final Validation Accuracy | **94.22%** |

---

## 7. Activation & Mathematical Stability

- **ReLU**: Enables the network to learn non-linear ergonomic decision boundaries (e.g., "risk increases non-linearly past 45° shoulder abduction").
- **Z-score Normalization**: Both inputs and outputs are normalized before training and de-normalized at inference, preventing large-gradient instability across the 4 output heads.
- **Feature Normalization at Inference**: Stored `X_mean` and `X_std` from training are applied to live camera data, ensuring the inference distribution matches the training distribution.

---

*Documented by ErgoVision AI Team · 2026*
