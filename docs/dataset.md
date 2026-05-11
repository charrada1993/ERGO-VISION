# Synthetic-Ergo-3D: Dataset Methodology

Training an AI for medical-grade ergonomics requires thousands of accurately labeled frames. Since public datasets rarely include the specific sub-scores required for RULA/REBA, we use a **Synthetic Bootstrap** methodology.

## 1. How Data is Generated
The `ai/synthetic_gen.py` script acts as a "Virtual Human" simulator. It generates 15,000+ postures by following these rules:

### A. Random Motion Sampling
The simulator picks random angles for every joint within human physiological limits:
- **Neck**: Flexion/Extension (-10° to +60°), Lateral (±35°).
- **Trunk**: Flexion/Extension (-10° to +90°).
- **Shoulders**: Abduction (0° to 180°).

### B. Landmark Projection
For every set of angles, the system calculates the **3D XYZ coordinates** of all 33 MediaPipe landmarks. This creates a perfect "Ground Truth" where the computer knows *exactly* what landmark configuration corresponds to what angle.

### C. Automatic Labeling (The Ground Truth)
Once the 3D skeleton is generated, it is passed through the **Official RULA/REBA Scoring Engine**. This assigns a score (1-7 for RULA, 1-15 for REBA) to that specific frame. 
- The AI then learns: **"When I see these landmarks, the score is X."**

## 2. Advantages of Synthetic Data
- **No Manual Labeling**: You don't need a doctor to manually grade thousands of photos.
- **Extreme Posture Coverage**: The simulator can generate "Impossible" or "High-Risk" postures that would be painful for a human to hold for data collection.
- **Perfect Accuracy**: Since the data is generated from math, there is zero human error in the training labels.
