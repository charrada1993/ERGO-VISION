# AI Operation: Inference-Only Mode

This folder contains the **Operational AI** engine. This is the version of the AI used in the live dashboard.

## AI Type: Static Inference Engine (SIE)
This is an "Operational-Only" AI. Unlike the training scripts, this code cannot learn or change. It is designed for **Reliability, Speed, and Stability**.

### Key Characteristics:
1.  **Zero Overhead**: No training logic, no gradient calculation, no backpropagation.
2.  **Deterministic**: For the same posture, it will always give the exact same RULA/REBA score.
3.  **Dependency-Lite**: Only requires `numpy`. This makes it immune to library version conflicts.
4.  **Hardware-Agnostic**: While optimized for Jetson, it can run on any ARM/x86 CPU with high efficiency.

## How it works:
The `inference.py` script loads the "Pre-Computed Brain" (`ergo_net_numpy.pkl`) and performs a simple 2-layer matrix multiplication. This is the "Operation Type" AI that powers the real-time tracking on your OAK-D camera feed.
