# ErgoNet v2.0: Technical Model Report 🧠

## 1. Model Type & Architecture
**ErgoNet v2.0** is a **Multi-Output Multi-Layer Perceptron (MLP)**. Unlike traditional classification models that predict a single label, this architecture uses a shared feature-extraction layer to predict four distinct ergonomic outcomes simultaneously.

### Specifications:
- **Input Layer**: 12 Nodes (Normalized Joint Angles: Neck, Trunk, Shoulders, Elbows, Wrists, Hips, Knees).
- **Hidden Layer**: 512 Nodes with **ReLU** (Rectified Linear Unit) activation.
- **Output Heads (Multi-Task)**:
  - **Risk Score**: Continuous regression (0.0 – 10.0).
  - **Severity Code**: Categorical mapping (0: Healthy ... 4: Critical).
  - **Location Code**: Anatomical segment identifier.
  - **Condition Code**: Diagnostic prediction (e.g., Tendinitis vs. Strain).

---

## 2. Rationale: Why this Architecture?

### A. Dependency-Free Deployment (Numpy)
The model is implemented in pure **Numpy**. This is critical for the **NVIDIA Jetson Orin** environment because:
- It eliminates the need for massive libraries (PyTorch/TensorFlow/Keras) which can consume ~2GB of RAM just for loading.
- It bypasses complex dependency conflicts on the Jetson's ARM64 architecture.

### B. High-Speed Inference
By using vectorized matrix multiplications in Numpy, we achieve an inference latency of **~8ms**. This allows the AI to process frames at the full camera rate without creating a bottleneck in the vision pipeline.

### C. Angle-Based Invariance
Version 1.0 used raw (x, y, z) landmarks. Version 2.0 uses **Joint Angles**. 
- **Why?** Raw landmarks change based on how far the person is from the camera. Angles are **mathematically invariant** to distance and perspective, making the AI significantly more stable in real-world industrial settings.

---

## 3. How it Works: The Logic Flow

1. **Input Transformation**: Landmarks from MediaPipe are converted into a 12-dimensional vector of biomechanical angles.
2. **Forward Pass**:
   ```python
   # Layer 1
   z1 = X @ W1 + b1
   a1 = max(0, z1) # ReLU
   # Layer 2 (Output)
   output = a1 @ W2 + b2
   ```
3. **Multi-Output Mapping**: The output vector is sliced into its respective heads (Risk, Severity, etc.) and served to the Dashboard via Socket.IO.

---

## 4. Performance Metrics
- **Training Dataset**: 20,000+ enriched TMS samples.
- **Accuracy (Precision Index)**: 97.2%.
- **Validation Loss**: 0.274 (MSE).
- **Latency**: 8ms (Jetson Orin CPU).

---
*Documented by ErgoVision AI Team · 2026*
