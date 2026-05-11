# AI Architecture: Ergo-ML-Core Deep Dive

The **Ergo-ML-Core** is a custom-built Neural Network designed to operate in high-performance environments like the NVIDIA Jetson Orin Nano. Unlike standard models that rely on heavy frameworks (PyTorch/TensorFlow), this engine is implemented in **Pure Numpy** to minimize latency and memory overhead.

## 1. The Multi-Head Design
Traditional pose estimation models only output joint coordinates. Ergo-ML-Core is different: it is a **Multi-Head Regressor**. This means a single forward pass through the network produces three distinct types of data:

### Head A: Biometric Angles (24 Outputs)
This head predicts the exact degrees of flexion, extension, lateral bending, and rotation.
- **Accuracy**: By training on synthetic metric data, it avoids the perspective distortion common in simple camera math.
- **Scope**: Covers Neck, Trunk, Shoulders, Elbows, and Wrists.

### Head B: Ergonomic Table Mapping (80 Outputs)
RULA and REBA are based on complex lookup tables. Instead of using `if/else` logic, this head learns the continuous relationship between posture and risk.
- **Group A (Upper Limb)**: Predicts scores for the upper arm, lower arm, wrist, and wrist twist.
- **Group B (Neck, Trunk, Leg)**: Predicts scores for neck flexion, trunk lean, and leg stability.

### Head C: Anomaly Probability (10 Outputs)
This head identifies if a specific body segment is in an "Anomalous State" (statistically extreme posture).
- **Output**: A value between 0.0 and 1.0 for each segment (e.g., "Left Shoulder: 0.95" indicates a severe anomaly).

## 2. Activation & Mathematical Stability
- **ReLU (Rectified Linear Unit)**: Used in the hidden layer to allow the network to learn non-linear ergonomic patterns.
- **Xavier Initialization**: Weights are initialized to prevent gradient vanishing/explosion, ensuring the model converges during training on the Jetson.
- **Feature Normalization**: Landmarks are Z-score normalized before processing to ensure that a person standing 1m away and a person standing 3m away result in the same ergonomic score.
