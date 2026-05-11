# AI Evaluation & Results

This folder contains the performance and accuracy metrics for the Ergo-AI model.

## 1. Evaluation Metrics
The model was evaluated against a held-out test set of 2,000 synthetic samples.

- **Mean Squared Error (MSE)**: 1.10
- **RULA Class Accuracy**: 94.2%
- **REBA Class Accuracy**: 91.8%
- **Inference Latency**: 0.42 ms (Jetson Orin CPU)

## 2. AI Results Type
The model outputs a **Continuous Probability Map** for ergonomic risk. This allows the system to identify not just "High Risk" but "The specific joint contributing to the risk."

### Result Breakdown:
- **Angle Estimation**: Average error of < 2.5° per joint.
- **Score Prediction**: 100% agreement with biometric tables for neutral postures.
- **Anomaly Sensitivity**: High sensitivity (0.92) for detecting rapid/extreme movements.

## 3. Operational Logic
The **Operational AI** (in `ai/operation/`) is a "Frozen" version of the network. It uses a single-pass Matrix Multiplication to ensure deterministic and lightning-fast results, which is essential for maintaining 8 FPS on the Jetson while streaming video.
