# ERGO-VISION: Advanced AI Documentation

This document explains the custom AI architecture and dataset logic used for automated RULA/REBA assessment on the **NVIDIA Jetson Orin (reComputer J3011)**.

## 1. AI Architecture: "Ergo-ML-Core"

Since the environment was optimized for performance and lacked heavy dependencies (like PyTorch or TensorFlow), I implemented a **Neural Network from scratch in raw Numpy**.

### Network Topology
- **Input Layer**: 99 Nodes (33 MediaPipe Landmarks x 3 coordinates [x, y, z]).
- **Hidden Layer**: 256 Nodes (ReLU Activation).
- **Output Layer**: 104 Nodes (Multi-Head Regression).
- **Optimization**: Multi-threaded BLAS (using Jetson's ARM v8.2 cores).

### Predictive Heads
The model uses a **Multi-Output** strategy to predict:
1.  **Motion Parameters**: 24 outputs for flexion, lateral, and rotation angles.
2.  **Ergonomic Scores**: 80 outputs representing the sub-scores from every RULA and REBA table.
3.  **Anomalies**: Probability scores for risk in specific body segments.

---

## 2. Dataset Type: "Synthetic-Ergo-3D"

Because professional ergonomic datasets are often private or restricted, the model was trained using a **High-Fidelity Synthetic Dataset** generated locally.

### Data Generation Logic
- **Kinematic Constraints**: The dataset simulates human movement within anatomical limits (e.g., Neck flexion: -10° to 60°).
- **Coordinate Space**: 3D landmark configurations are generated to match the MediaPipe coordinate system (X: Right, Y: Down, Z: Forward).
- **Labeling**: "Ground Truth" RULA/REBA scores are calculated for every frame using the official biometric tables.
- **Volume**: 15,000 unique samples, including "Anomalous" (extreme) and "Safe" (neutral) postures.

---

## Detailed Documentation Modules

For a deeper explanation of each component, please refer to the following files in the `docs/` folder:

1.  **[AI Architecture](docs/architecture.md)**: Deep dive into the Numpy Multi-Head Neural Network.
2.  **[Dataset & Methodology](docs/dataset.md)**: Explaining the Synthetic-Ergo-3D data generation.
3.  **[10-Day Forecasting](docs/forecasting.md)**: The math behind the LSTM time-series prediction.
4.  **[Jetson Hardware Tuning](docs/jetson_optimization.md)**: Power and performance optimizations for the Orin platform.
