# ERGO-VISION 🦺

> **Real-time ergonomic posture assessment system** powered by OAK-D depth cameras, MediaPipe pose estimation, and **ErgoNet v2.0 Neural Engine** — trained to **97.14% accuracy** on 20,000+ TMS samples for clinical-grade musculoskeletal diagnostics.

*Last updated: 2026-05-29 — Full RPY joint angle computation + live dashboard chart fix*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [System Architecture](#system-architecture)
4. [Features](#features)
5. [Hardware Requirements](#hardware-requirements)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Joint Angles Computed](#joint-angles-computed)
10. [API Reference](#api-reference)
11. [RULA / REBA Scoring Reference](#rula--reba-scoring-reference)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)

---

## Overview

**ERGO-VISION** is a fully open-source, real-time ergonomic risk assessment platform for industrial and occupational health environments. It uses OAK-D depth cameras to capture synchronized RGB + stereo depth streams, runs MediaPipe pose estimation, computes 30+ joint angles (including full Roll/Pitch/Yaw per joint), and streams everything to a live web dashboard with Chart.js real-time sparklines.

---

## Technology Stack

### 🖥️ Hardware Platform

| Component | Technology | Role |
|---|---|---|
| Edge AI Computer | **NVIDIA Jetson Orin** (reComputer J3011) | Main inference host (ARM Cortex-A78AE + CUDA) |
| Depth Camera | **Luxonis OAK-D** (OpenCV AI Kit with Depth) | RGB 1280×720 + aligned stereo depth |
| USB | **USB 2.0 / USB 3.0** | Camera data transport via XLink protocol |

---

### 🐍 Backend — Python

| Library | Version | Role |
|---|---|---|
| **Python** | 3.10+ | Primary language |
| **depthai** | 2.24+ | OAK-D camera SDK — pipeline, XLink queues, depth post-processing |
| **mediapipe** | 0.10+ | Human pose estimation — 33 body keypoints (x, y, z) |
| **opencv-python** | 4.8+ | Image processing, Lucas-Kanade optical flow, MJPEG streaming |
| **numpy** | 1.24+ | All 3D vector maths, angle computation, EMA smoothing |
| **flask** | 3.0+ | Web server and HTTP routes |
| **flask-socketio** | 5.3+ | Real-time WebSocket communication (Socket.IO v4) |
| **simple-websocket** | — | WebSocket backend transport for Flask-SocketIO |
| **pandas** | 2.0+ | Session CSV logging and data export |
| **matplotlib** | 3.7+ | Time-series chart generation for PDF reports |
| **reportlab** | 4.0+ | PDF ergonomic report builder |

---

### 🌐 Frontend — Web Dashboard

| Technology | Version | Role |
|---|---|---|
| **HTML5** | — | Page structure and semantic layout |
| **Vanilla CSS3** | — | Glassmorphic dark theme, CSS custom properties (design tokens) |
| **JavaScript (ES2020)** | — | All client-side logic, DOM manipulation |
| **Socket.IO Client** | 4.5.0 | Real-time WebSocket data reception from Flask backend |
| **Chart.js** | 4.4.0 | Live rolling line charts (joint angles, RULA/REBA scores, 12 sparklines) |
| **Three.js** | r128 | Interactive 3D skeleton viewer at `/3d` |
| **Font Awesome** | 6.0 | UI icons throughout the dashboard |
| **Google Fonts** | — | Inter / JetBrains Mono typography |

---

### 🧠 AI / Machine Learning

| Technology | Role |
|---|---|
| **ErgoNet v2.0** | Custom 4-head MLP (angle-based): risk score, severity, location code, condition code |
| **NumPy MLP** | Dependency-free inference engine (NEON-optimised on ARM) — no TensorFlow/PyTorch needed |
| **Synthetic Data Generator** | `ai/synthetic_gen.py` — high-fidelity TMS ergonomic dataset fabrication (20,000+ samples) |
| **EMA Temporal Smoothing** | α=0.15 — 85% history weight, eliminates keypoint jitter at RULA scoring thresholds |
| **Score Hysteresis** | Prevents RULA/REBA flickering across risk-level boundaries |
| **Dropout Holdout** | Holds last-good angles for up to 6 missed frames (~0.75 s) during tracking loss |

**ErgoNet v2.0 Architecture:**
```
Input (12 joint angles) → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU)
→ 4 output heads:
    risk_score     (continuous 0–10)
    severity_code  (class 0–4: Healthy → Critical)
    location_code  (class 0–8: body region)
    condition_code (class 0–17: MSK condition)
```

**Training results:** Loss `0.2742` | Train Acc `97.14%` | Val Acc `94.22%`

---

### 📐 Biomechanics / Ergonomics Standards

| Standard | Implementation |
|---|---|
| **RULA** (Rapid Upper Limb Assessment) | Full 7-level score via exact official lookup tables |
| **REBA** (Rapid Entire Body Assessment) | Full 15-level score via exact official lookup tables |
| **McAtamney & Corlett (1993)** | RULA methodology reference |
| **Hignett & McAtamney (2000)** | REBA methodology reference |

---

### ⚙️ DevOps / Deployment

| Technology | Role |
|---|---|
| **Bash** (`run.sh`) | Jetson startup script: nvpmodel MAXN, jetson_clocks, USB power management, log rotation |
| **taskset** | CPU affinity pinning to cores 0–3 (Cortex-A78 big cores) |
| **Git / GitHub** | Version control and remote repository |
| **MJPEG over HTTP** | Video streaming endpoint (`/video_feed`, `/depth_feed`) at 8 fps |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OAK-D Camera (USB 2.0)                     │
│         RGB 1280×720 + Stereo Depth aligned @ 8 fps         │
└────────────────────────┬────────────────────────────────────┘
                         │  DepthAI XLink
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  camera/manager.py  —  DepthAI Pipeline                     │
│  • Non-blocking queues (size=1)  • tryGetAll() freshest     │
│  • Depth: speckle + temporal + spatial + threshold filters  │
│  • Simulation mode (MockCamera) when no OAK-D present       │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐
│  pose/estimator.py        │  MediaPipe Pose → 33 keypoints
│  pose/fusion.py           │  Multi-camera landmark merge
│  pose/skeleton.py         │  3D vector maths → 30+ angles
│  (3×3 median depth patch) │  EMA α=0.15 + dropout holdout
└───────────────┬───────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌────────────┐   ┌────────────┐
│ rula.py    │   │ reba.py    │
│ Score 1–7  │   │ Score 1–15 │
└────────────┘   └────────────┘
       │                 │
       └────────┬────────┘
                ▼
┌─────────────────────────────────────────────────────────────┐
│  web/socket_events.py  — Background Thread @ ~8 Hz          │
│  ai/operation/inference.py — ErgoNet v2.0 inference         │
│  Emits: pose_update, skeleton_3d via Socket.IO              │
└────────────────────────┬────────────────────────────────────┘
                         │  WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Flask Web Dashboard (web/routes.py)                        │
│  ├── / dashboard    — Chart.js live charts + body map       │
│  ├── /camera        — MJPEG video + depth feed              │
│  ├── /rula          — RULA sub-score breakdown              │
│  ├── /reba          — REBA sub-score breakdown              │
│  ├── /3d            — Three.js 3D skeleton viewer           │
│  ├── /ai            — ErgoNet training curves + inference   │
│  ├── /collection    — CSV session recorder                  │
│  └── /report        — PDF report generator                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| 🎥 **OAK-D Depth Camera** | RGB + stereo depth, aligned, post-processed |
| 🦴 **MediaPipe Pose** | 33 body keypoints, CPU-only, real-time |
| 📐 **Full RPY Angles** | Pitch, Roll, Yaw computed per joint (neck, trunk, elbow, wrist, thigh) |
| 📊 **RULA / REBA** | Official lookup-table scoring, Groups A+B, sub-scores |
| 🧠 **ErgoNet v2.0** | Angle-based MLP, 97.14% accuracy, 18 MSK condition classes |
| 📈 **12 Live Sparklines** | Per-joint trend charts with live RPY numeric badges |
| 🌀 **Visual IMU** | Roll/Pitch/Yaw from Lucas-Kanade optical flow |
| 🌐 **Socket.IO Dashboard** | Real-time WebSocket streaming to any browser |
| 🎬 **MJPEG Streams** | Live RGB + depth colourmap at 8 fps over HTTP |
| 📄 **PDF Reports** | Automated ergonomic risk reports from CSV sessions |
| ⚡ **Jetson Optimised** | CPU affinity, MAXN mode, non-blocking queues, 8 fps cap |
| 🌙 **Dark / Light Mode** | Theme toggle with Chart.js theme sync |
| 🔄 **Simulation Mode** | MockCamera when no OAK-D attached |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Camera | 1× OAK-D Lite | 1–3× OAK-D with BNO086 IMU |
| Host | 4-core ARM/x86, 4 GB RAM | NVIDIA Jetson Orin, 8 GB RAM |
| USB | USB 2.0 | USB 3.0 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

---

## Project Structure

```
ERGO-VISION/
├── app.py                      # Entry point — orchestrates all modules
├── config.py                   # Global config (FPS, paths, thresholds)
├── requirements.txt            # Python dependencies
├── run.sh                      # Jetson startup script
│
├── camera/
│   ├── manager.py              # OAK-D DepthAI pipeline + MockCamera
│   ├── imu_manager.py          # Visual IMU via Lucas-Kanade optical flow
│   └── calibration.py          # RGB intrinsics / extrinsics
│
├── pose/
│   ├── estimator.py            # MediaPipe Pose → 33 landmarks
│   ├── fusion.py               # Multi-camera landmark fusion
│   └── skeleton.py             # 3D joint angle computation (30+ keys)
│
├── ergonomics/
│   ├── rula.py                 # RULA calculator (score 1–7)
│   ├── reba.py                 # REBA calculator (score 1–15)
│   └── risk.py                 # Anomaly detector
│
├── ai/
│   ├── models/ergo_net_v2.pkl  # ErgoNet v2.0 weights
│   ├── data/                   # TMS dataset + training logs
│   ├── operation/              # Inference engine + ONNX export
│   ├── train_v2.py             # Training pipeline
│   └── synthetic_gen.py        # Synthetic data generator
│
├── web/
│   ├── routes.py               # Flask routes
│   ├── socket_events.py        # Socket.IO + processing thread
│   ├── static/css/style.css    # Glassmorphic dark/light theme
│   └── static/js/
│       ├── dashboard.js        # Chart.js charts + Socket.IO client
│       └── 3d_viewer.js        # Three.js 3D skeleton
│   └── templates/              # Jinja2 HTML pages
│
├── reporting/
│   ├── report_generator.py     # PDF builder (ReportLab)
│   └── graphs.py               # Matplotlib chart helpers
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── model_report.md
    ├── jetson_optimization.md
    └── onnx_guide.md
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/charrada1993/ERGO-VISION.git
cd ERGO-VISION

# 2. Create virtualenv
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Linux) OAK-D udev rules
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Quick Start

```bash
# Option A — Jetson launcher (pins clocks, manages USB power)
bash run.sh

# Option B — Direct Python
python3 app.py
```

Open **http://localhost:5000** (or your Jetson IP on port 5000 from any device on the network).

| URL | Page |
|---|---|
| `/` | Main dashboard — live charts, body risk map, AI analysis |
| `/camera` | RGB + depth MJPEG feed |
| `/rula` | RULA sub-score breakdown |
| `/reba` | REBA sub-score breakdown |
| `/3d` | Three.js interactive 3D skeleton |
| `/ai` | ErgoNet training curves + live inference monitor |
| `/collection` | CSV session recorder |
| `/report` | Automated PDF report generator |

---

## Joint Angles Computed

`pose/skeleton.py` emits the following angle keys via `pose_update`:

| Key | Description |
|---|---|
| `neck` | Neck flexion/extension (Pitch) |
| `neck_roll` | Head lateral tilt (Roll) |
| `neck_yaw` | Head left/right rotation (Yaw) |
| `trunk` | Trunk forward lean (Pitch) |
| `trunk_roll` | Trunk lateral tilt (Roll) |
| `trunk_yaw` | Trunk rotation — shoulder vs hip line (Yaw) |
| `upper_arm_left/right` | Shoulder elevation |
| `abd_l/r` | Shoulder abduction |
| `elbow_left/right` | Elbow interior angle |
| `elb_roll_l/r` | Forearm pronation/supination (Roll) |
| `wrist_left/right` | Wrist flexion (Pitch) |
| `wri_roll_l/r` | Wrist radial/ulnar deviation (Roll) |
| `wri_yaw_l/r` | Wrist twist in transverse plane (Yaw) |
| `hip_left/right` | Hip flexion |
| `thi_roll_l/r` | Thigh abduction in frontal plane (Roll) |
| `thi_yaw_l/r` | Hip internal rotation in horizontal plane (Yaw) |
| `knee_left/right` | Knee flexion |

---

## API Reference

### Socket.IO Events (Server → Client)

| Event | Description |
|---|---|
| `config` | `{mode, usb3}` — camera count and USB speed |
| `pose_update` | Full posture payload at ~8 Hz |
| `skeleton_3d` | `{landmarks: [[x,y,z]×33]}` for Three.js viewer |

**`pose_update` payload:**
```json
{
  "angles":       {"neck": 12.3, "neck_yaw": -5.1, "trunk": 8.2, ...},
  "rula":         4,
  "reba":         7,
  "risk_level":   "Medium",
  "anomalies":    ["Neck flexion: 42.0° (>40°)"],
  "rula_details": {"score_a": 3, "score_b": 4, "score_c": 4},
  "reba_details": {"table_a": 5, "table_b": 6, "score_c": 7},
  "ai_results":   {"risk_score": 3.2, "severity_code": 2, "condition_code": 9}
}
```

### REST Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/config` | GET | Camera mode and hardware config |
| `/api/sessions` | GET | List recorded CSV sessions |
| `/api/reports` | GET | List generated PDF reports |
| `/api/download_csv/<file>` | GET | Download a session CSV |
| `/api/download_report/<file>` | GET | Download a PDF report |
| `/api/generate_report/<file>` | GET | Generate PDF from CSV |
| `/api/training_log` | GET | ErgoNet v2.0 training history JSON |
| `/video_feed` | GET | MJPEG RGB stream (8 fps) |
| `/depth_feed` | GET | MJPEG depth colourmap stream (4 fps) |

---

## RULA / REBA Scoring Reference

### RULA Final Score

| Score | Risk | Action |
|---|---|---|
| 1–2 | ✅ Acceptable | No action |
| 3–4 | ⚠️ Low | Monitor |
| 5–6 | 🔶 Medium | Change needed |
| 7 | 🔴 Very High | Immediate action |

### REBA Final Score

| Score | Risk | Action |
|---|---|---|
| 1 | ✅ Negligible | No action |
| 2–3 | ⚠️ Low | Monitor |
| 4–7 | 🔶 Medium | Improvement needed |
| 8–10 | 🔴 High | Rapid intervention |
| 11–15 | 🆘 Very High | Immediate action |

---

## Configuration

Edit `config.py`:

```python
class Config:
    MAX_CAMERAS      = 3      # Maximum OAK-D cameras
    PROCESSING_FPS   = 10     # Pose estimation rate (Hz)
    LOG_INTERVAL     = 0.5    # CSV logging interval (s)
    LOAD_KG_DEFAULT  = 0      # Carried load (kg)

class JetsonConfig:
    VIDEO_WIDTH      = 320    # MJPEG stream width
    VIDEO_HEIGHT     = 240    # MJPEG stream height
    VIDEO_STREAM_FPS = 8      # MJPEG FPS cap
    VIDEO_JPEG_QUALITY = 50   # JPEG quality (0–100)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No OAK-D device found` | Check USB, run `lsusb \| grep 03e7`, install udev rules |
| `Port 5000 already in use` | Run `fuser -k 5000/tcp` then restart |
| Charts blank / no data | Check browser console for JS errors; ensure camera is connected |
| Blank video feed | Camera in simulation mode — connect OAK-D |
| PDF not generated | Ensure `reports/` is writable; `pip install reportlab matplotlib` |
| MediaPipe no landmarks | Improve lighting; subject must be fully visible in frame |

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

**MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Luxonis DepthAI](https://docs.luxonis.com/) — OAK-D camera SDK
- [Google MediaPipe](https://developers.google.com/mediapipe) — Human pose estimation
- [Chart.js](https://www.chartjs.org/) — Real-time data visualization
- [Three.js](https://threejs.org/) — 3D skeleton viewer
- McAtamney & Corlett (1993) — RULA methodology
- Hignett & McAtamney (2000) — REBA methodology

---

*Built with ❤️ for occupational health and safety on NVIDIA Jetson Orin.*
