# ERGO-VISION: Advanced AI Documentation

*Last updated: 2026-05-11 — ErgoNet v2.0 training complete.*

This document explains the custom AI architecture, dataset logic, and training pipeline used for automated TMS (Musculoskeletal Disorder) risk assessment on the **NVIDIA Jetson Orin (reComputer J3011)**.

---

## 1. AI Architecture: ErgoNet v2.0

Since the deployment environment requires minimal dependencies and maximum speed, the AI is implemented as a **Neural Network from scratch in raw NumPy**.

### Network Topology

| Layer | Shape | Activation |
|---|---|---|
| Input | 12 joint angles | Z-score normalized |
| Hidden | 512 nodes | ReLU |
| Output | 4 diagnostic heads | Linear (regression) |

### Predictive Heads (Multi-Task Output)

| Head | Output | Description |
|---|---|---|
| `risk_score` | Float 0.0–10.0 | Overall ergonomic risk magnitude |
| `severity_code` | Int 0–4 | Severity level: Healthy → Critical |
| `location_code` | Int | Anatomical region of highest risk |
| `condition_code` | Int | TMS condition (Tendinitis, Strain, etc.) |

### Key Improvements in v2.0

- **Angle-Based Inputs** (replaces raw landmarks): joint angles are scale-invariant and camera-distance-invariant, greatly improving real-world stability.
- **512-node hidden layer** (vs. 256 in v1): higher capacity for multi-joint interaction patterns.
- **TMS Enriched Dataset**: 20,000+ samples vs. 15,000 in v1, with condition-specific data augmentation.

---

## 2. Training Results (v2.0 — 2026-05-11)

| Metric | Value |
|---|---|
| Dataset | `dataset_TMS_enriched.csv` (~20,000 samples) |
| Epochs | 500 |
| Learning Rate | 0.005 |
| **Final Training Loss (MSE)** | **0.2742** |
| **Final Training Accuracy** | **97.14%** |
| **Final Validation Accuracy** | **94.22%** |
| Model file | `ai/models/ergo_net_v2.pkl` |

---

## 3. Dataset: Synthetic-Ergo-3D (TMS Enriched)

Because professional ergonomic datasets with precise clinical labels are rare, the model is trained on a **High-Fidelity Synthetic Dataset** generated via `ai/synthetic_gen.py`.

### Generation Logic

- **Kinematic Constraints**: Simulates human movement within anatomical limits (e.g., Neck: −10° to +60°).
- **Coordinate Space**: 3D landmark configurations match the MediaPipe coordinate system.
- **Auto-Labeling**: Ground-truth RULA/REBA scores computed for every frame using official tables.
- **TMS Enrichment**: Over-represents high-risk postures and condition-specific patterns for clinical sensitivity.

---

## 4. Detailed Documentation

For deeper technical explanations, see the `docs/` folder:

| File | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Network topology, forward pass math, initialization |
| [docs/dataset.md](docs/dataset.md) | TMS enriched dataset structure and generation methodology |
| [docs/model_report.md](docs/model_report.md) | Full training results, metrics, and saved model schema |
| [docs/forecasting.md](docs/forecasting.md) | 10-day LSTM risk forecasting logic |
| [docs/jetson_optimization.md](docs/jetson_optimization.md) | Hardware tuning, power mode, memory optimization |
| [docs/onnx_guide.md](docs/onnx_guide.md) | Future ONNX → TensorRT export pathway |
| [ai/evaluation/README.md](ai/evaluation/README.md) | Epoch-by-epoch training progression and inference benchmarks |
| [ai/operation/README.md](ai/operation/README.md) | Operational inference engine details |

---

*Documented by ErgoVision AI Team · 2026*
