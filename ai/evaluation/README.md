# AI Evaluation & Results — ErgoNet v2.0

*Last updated: 2026-05-11 — Post training run (500 epochs, `dataset_TMS_enriched.csv`)*

---

## 1. Training Configuration

| Parameter | Value |
|---|---|
| **Script** | `ai/train_v2.py` |
| **Dataset** | `ai/data/dataset_TMS_enriched.csv` (~20,000 samples) |
| **Input Features** | 12 joint angles (bilateral, Z-score normalized) |
| **Output Heads** | 4 (risk_score, severity_code, location_code, condition_code) |
| **Architecture** | MLP: 12 → 512 (ReLU) → 4 |
| **Optimizer** | Gradient Descent, lr = 0.005 |
| **Epochs** | 500 |
| **Loss Function** | Mean Squared Error (MSE) |

---

## 2. Training Results (Completed Run)

| Metric | Value |
|---|---|
| **Final Training Loss (MSE)** | **0.2742** |
| **Final Training Accuracy** | **97.14%** |
| **Final Validation Loss** | **0.2971** |
| **Final Validation Accuracy** | **94.22%** |
| **Model File** | `ai/models/ergo_net_v2.pkl` |
| **Training Log** | `ai/data/training_log.json` |

### Epoch Progression Highlights

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 1 | ~0.843 | ~0.851 | ~0.826 |
| 100 | ~0.322 | ~0.935 | ~0.907 |
| 200 | ~0.303 | ~0.950 | ~0.921 |
| 300 | ~0.290 | ~0.957 | ~0.929 |
| 400 | ~0.281 | ~0.963 | ~0.934 |
| **500** | **0.274** | **0.971** | **0.942** |

---

## 3. Inference Performance

| Metric | Value |
|---|---|
| **Inference Latency** | ~8 ms (Jetson Orin CPU) |
| **RAM Footprint** | ~15 MB (model + buffers) |
| **Dependencies** | `numpy` only |
| **Throughput** | Compatible with 30 FPS camera feed |

---

## 4. Diagnostic Output Quality

| Output Head | Description | Quality |
|---|---|---|
| `risk_score` | 0.0–10.0 continuous risk magnitude | Smooth, well-calibrated |
| `severity_code` | 0–4 categorical severity | High agreement with RULA/REBA reference |
| `location_code` | Anatomical region | Correctly identifies dominant risk joint |
| `condition_code` | TMS condition type | Strong on asymmetric patterns |

---

## 5. Operational Logic

The **Operational AI** (`ai/operation/`) is the "frozen" production inference engine. It:
1. Loads the pre-trained `ergo_net_v2.pkl` at server startup.
2. Accepts a 12-element angle vector from `pose/skeleton.py`.
3. Applies stored `X_mean`/`X_std` normalization.
4. Runs a single 2-layer forward pass (< 8 ms).
5. De-normalizes output using stored `y_mean`/`y_std`.
6. Emits results via Socket.IO to the `/ai` dashboard page.

This deterministic, zero-learning inference ensures **100% reproducibility** for the same input posture across all sessions.

---

*Documented by ErgoVision AI Team · 2026*
