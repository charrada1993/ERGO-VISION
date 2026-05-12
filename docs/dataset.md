# Dataset: TMS Enriched Ergonomic Dataset

*Last updated: 2026-05-11*

ErgoNet v2.0 is trained on the **TMS Enriched Dataset** (`ai/data/dataset_TMS_enriched.csv`), a high-fidelity synthetic ergonomic dataset generated specifically for musculoskeletal disorder (TMS) risk prediction.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **File** | `ai/data/dataset_TMS_enriched.csv` |
| **Size** | ~18.6 MB |
| **Samples** | 20,000+ unique posture records |
| **Input Features** | 12 joint angles (bilateral, biomechanical) |
| **Target Features** | 4 clinical outputs (risk, severity, location, condition) |

---

## 2. Input Features (12 Angle Columns)

All angles are in **degrees** and represent real biomechanical joint measurements:

| Column Name | Joint | Range |
|---|---|---|
| `Neck_Flexion_deg` | Cervical flexion/extension | −10° to +60° |
| `Trunk_Flexion_deg` | Lumbar forward lean | −10° to +90° |
| `R_Shoulder_Flexion_deg` | Right shoulder elevation | 0° to 180° |
| `L_Shoulder_Flexion_deg` | Left shoulder elevation | 0° to 180° |
| `R_Elbow_Flexion_deg` | Right elbow | 0° to 150° |
| `L_Elbow_Flexion_deg` | Left elbow | 0° to 150° |
| `R_Wrist_Deviation_deg` | Right wrist radial/ulnar | −30° to +30° |
| `L_Wrist_Deviation_deg` | Left wrist radial/ulnar | −30° to +30° |
| `R_Hip_Flexion_deg` | Right hip | 0° to 120° |
| `L_Hip_Flexion_deg` | Left hip | 0° to 120° |
| `R_Knee_Flexion_deg` | Right knee | 0° to 135° |
| `L_Knee_Flexion_deg` | Left knee | 0° to 135° |

---

## 3. Target Features (4 Clinical Labels)

| Column Name | Type | Description |
|---|---|---|
| `risk_score` | Float (0.0–10.0) | Overall ergonomic risk magnitude |
| `severity_code` | Int (0–4) | Severity: 0=Healthy, 1=Low, 2=Moderate, 3=High, 4=Critical |
| `location_code` | Int | Anatomical region of highest risk |
| `condition_code` | Int | TMS condition: Tendinitis, Back Pain, Strain, etc. |

---

## 4. Synthetic Generation Methodology

Training-grade ergonomic datasets with precise clinical labels are rarely publicly available. ERGO-VISION uses a **Synthetic Bootstrap** approach via `ai/synthetic_gen.py`.

### A. Random Motion Sampling
The simulator samples random joint angles within anatomically valid ranges:
- **Neck**: Flexion/Extension (−10° to +60°), Lateral (±35°).
- **Trunk**: Flexion (−10° to +90°).
- **Shoulders**: Abduction/Flexion (0° to 180°).
- **Elbows**: Flexion (0° to 150°).
- **Wrists**: Deviation (±30°).
- **Hips/Knees**: Physiological range.

### B. Landmark Projection
For each angle set, the system computes the **3D XYZ coordinates** of all 33 MediaPipe landmarks, creating a perfect ground truth that links landmark geometry to precise angle values.

### C. Automatic RULA/REBA Labeling
Every generated posture is passed through the **official RULA/REBA scoring engine** to assign clinical scores. The AI then learns: *"When I see these angles, the risk state is X."*

### D. TMS Enrichment
The dataset is additionally enriched with **condition-specific patterns**:
- Over-represented high-risk postures (RULA 5+) to improve sensitivity.
- Bilateral asymmetry patterns (left/right imbalance) associated with real-world TMS development.
- Repetitive micro-posture variations to improve generalization.

---

## 5. Advantages of Synthetic Data

| Advantage | Details |
|---|---|
| **No Manual Labeling** | No need for a clinician to grade thousands of images |
| **Extreme Coverage** | Can generate postures too painful for humans to hold during data collection |
| **Zero Label Error** | Labels are computed from math — no human annotation mistakes |
| **Controllable Distribution** | Can over-sample rare high-risk conditions as needed |
| **Privacy Compliant** | No real patient or worker data involved |

---

## 6. Preprocessing in Training

Before training, the following normalization is applied:

```python
# Input normalization (Z-score)
X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-6
X_norm = (X - X_mean) / X_std

# Output normalization (Z-score)
y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
y_norm = (y - y_mean) / y_std
```

The `X_mean`, `X_std`, `y_mean`, and `y_std` statistics are saved inside the model `.pkl` file and applied identically at inference time.

---

*Documented by ErgoVision AI Team · 2026*
