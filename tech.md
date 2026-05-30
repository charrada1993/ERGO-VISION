# ERGO-VISION Technology Stack 🛠️

This document cataloges all technologies, frameworks, hardware components, and software libraries used in the **ERGO-VISION** real-time ergonomic posture assessment system.

---

## 📋 Technology Stack Overview

```
                        ┌──────────────────────────────────────┐
                        │          NVIDIA Jetson Orin          │
                        │      (MAXN Power, CPU Affinity)      │
                        └──────────────────┬───────────────────┘
                                           │
        ┌───────────────────────────────────┼──────────────────────────────────┐
        ▼                                   ▼                                  ▼
┌──────────────┐                   ┌──────────────┐                   ┌──────────────┐
│  Hardware    │                   │   Backend    │                   │   Frontend   │
│  • OAK-D Cam │                   │  (Python 3)  │                   │  (JS/HTML5)  │
│  • USB 3.0   │                   │  • Flask     │                   │  • Vanilla   │
│              │                   │  • Socket.IO │                   │    CSS       │
└──────────────┘                   │  • MediaPipe │                   │  • Chart.js  │
                                   │  • NumPy MLP │                   │  • Three.js  │
                                   └──────────────┘                   └──────────────┘
```

---

## 🖥️ 1. Hardware Platform

| Technology | Role & Integration | Details |
|---|---|---|
| **NVIDIA Jetson Orin** (reComputer J3011) | **Main Host Computer** | ARM Cortex-A78AE CPU cores (ARM64 architecture) running NVIDIA JetPack Linux. Handles the real-time processing threads and Flask web server. |
| **Luxonis OAK-D / OAK-D Lite** | **Stereo Depth Sensing** | Captures synchronized RGB (1280×720) and aligned stereo depth streams. Performs hardware-accelerated spatial depth filtering. |
| **USB 3.0 / USB 2.0** | **Data Transport Protocol** | Facilitates communication between the host Jetson and OAK-D cameras using Luxonis's XLink protocol. |

---

## 🐍 2. Backend Engine (Python 3.10)

The backend is built as a modular Python application running on Jetson Linux. It handles OAK-D pipelines, pose landmark extraction, biomechanical angle calculations, AI inference, and server-side WebSocket communication.

### Core Libraries & Utilities
- **`depthai`**: Luxonis's DepthAI SDK. Used to interface with the OAK-D camera, configure input queues, pull synchronized frames, and enable non-blocking queue handlers.
- **`mediapipe`**: Google's MediaPipe framework (optimized ARM64 build). Extracts 33 3D body keypoints (landmarks) in real time from the RGB video feed.
- **`numpy`**: Performance vector mathematics. Powers all 3D biomechanical joint angle calculations, Exponential Moving Average (EMA) filtering, normalization, and dependency-free MLP inference.
- **`opencv-python-headless`**: OpenCV library. Used for video frame manipulation, Lucas-Kanade visual optical flow tracking, color-mapping depth matrices, and encoding MJPEG streams.
- **`pandas`**: Data logging and management. Saves real-time sessions into structured CSV files for analysis and reporting.
- **`reportlab`**: Clinical report compilation. Generates professional PDF ergonomic reports from historical telemetry.
- **`matplotlib`**: Dynamic chart generation. Generates time-series line plots of joint angles and RULA/REBA trends, which are embedded directly into PDF reports.

### Networking & Web Server
- **`flask`**: Micro web framework hosting REST endpoints and serving the dashboard views.
- **`flask-socketio`**: Enables low-latency, real-time bidirectional WebSocket communication (Socket.IO protocol v4) between the Python processing thread and the web browser.
- **`simple-websocket`**: Acts as the fast WebSocket transport server implementation underneath Flask-SocketIO.

---

## 🌐 3. Frontend UI Dashboard

The frontend is a lightweight, high-performance, single-page dashboard designed to run in any standard web browser (Chrome, Safari, Firefox, Edge). It avoids bulky frameworks to maintain maximum rendering speeds on low-power devices.

- **Semantic HTML5**: Structures pages like the main dashboard, AI training monitor, interactive 3D player, and reporting views.
- **Vanilla CSS3**: Styles the application using a customized glassmorphic design system. Features dark and light modes using CSS variables for theme management.
- **JavaScript (ES2020)**: Implements dynamic DOM manipulation, real-time telemetry rendering, theme swapping, and WebSocket event handling.
- **Socket.IO Client (v4.5.0)**: Subscribes to the backend's real-time events (`pose_update`, `skeleton_3d`, `config`) to dynamically update UI cards.
- **Chart.js (v4.4.0)**: Drives the real-time sparklines and time-series history charts. Renders rolling 30-second joint angle graphs with real-time numeric badges.
- **Three.js (r128)**: Renders a fully interactive 3D WebGL skeleton representation of the captured user landmarks at the `/3d` view.
- **Font Awesome (v6.0)**: Provides responsive vector icons throughout the interface.
- **Google Fonts (Inter & JetBrains Mono)**: Curated typography for readability.

---

## 🧠 4. Artificial Intelligence & Biomechanics

### ErgoNet v2.0 Neural Engine
- **Architecture**: A multi-headed Multi-Layer Perceptron (MLP) mapping 12 joint inputs to 4 diagnostic outputs:
  - **`risk_score`**: Continuous (0.0 to 10.0) severity magnitude.
  - **`severity_code`**: Categorical severity (Healthy, Low, Medium, High, Critical).
  - **`location_code`**: Anatomical region (e.g. neck, trunk, shoulder, wrist).
  - **`condition_code`**: 18 musculoskeletal disorders (e.g. Tech Neck, Carpal Tunnel, Lumbar Hernia).
- **NumPy MLP Engine**: To avoid heavy framework dependencies (TensorFlow/PyTorch) on the Jetson, inference is executed via a dependency-free forward-pass NumPy implementation, optimized via ARM NEON instructions.
- **ONNX Pipeline**: Model includes export scripts to ONNX format (`ai/operation/export_onnx.py`) to support future acceleration via TensorRT.

### Temporal Stabilization & Biomechanical Standards
- **EMA Filter**: Computes Exponential Moving Averages ($\alpha = 0.15$) to eliminate MediaPipe landmark jitter.
- **Hysteresis Logic**: Smooths RULA/REBA scoring updates across risk boundaries, eliminating screen-flickering.
- **Dropout Holdout**: Remembers the last-valid posture state for up to 6 frames during visual occlusion.
- **Biomechanical Formulas**: Implements exact official RULA and REBA assessment lookup tables.

---

## ⚙️ 5. DevOps & Host Optimization

To achieve a stable 8+ FPS pipeline on the Jetson Orin CPU:
- **`run.sh` Bash Automation**: Sets Jetson Power Mode to `MAXN` high-performance, runs `jetson_clocks` to lock CPU/GPU frequencies, disables USB autosuspend, and rotates log files.
- **`taskset` CPU Pinning**: Binds the critical real-time frame processing loop to CPU cores 0–3 (the A78AE big cores) to prevent scheduler context switching.
- **Log Rotation**: Limits log sizes to prevent disk overflow on edge installations.
