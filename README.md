# ERGO-VISION 🦴📷

> **Real-time ergonomic posture assessment system** using 1–3 OAK-D cameras, MediaPipe pose estimation, RULA/REBA scoring, Visual IMU, and a live web dashboard.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Hardware Requirements](#hardware-requirements)
5. [Software Requirements](#software-requirements)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Modules Documentation](#modules-documentation)
   - [Camera Manager](#camera-manager)
   - [Visual IMU](#visual-imu)
   - [Pose Estimation](#pose-estimation)
   - [Ergonomic Scoring (RULA & REBA)](#ergonomic-scoring-rula--reba)
   - [Web Dashboard](#web-dashboard)
   - [Data Logging & Reports](#data-logging--reports)
10. [RULA Scoring Reference](#rula-scoring-reference)
11. [REBA Scoring Reference](#reba-scoring-reference)
12. [Visual IMU Formulas](#visual-imu-formulas)
13. [Configuration](#configuration)
14. [API Reference](#api-reference)
15. [Troubleshooting](#troubleshooting)
16. [Contributing](#contributing)

---

## Overview

**ERGO-VISION** is a fully open-source, real-time ergonomic risk assessment system designed for industrial and occupational health environments. It uses one to three **OAK-D** (OpenCV AI Kit with Depth) cameras to:

- Capture synchronized **RGB + aligned stereo depth** streams
- Detect and track human body keypoints with **MediaPipe Pose** (Optimized for NVIDIA Jetson Orin)
- **Depth-Based Masking**: Auto-filters background workers for robust single-subject tracking
- Compute **joint angles** (neck, trunk, arms, wrists, elbows)
- Calculate standardized **RULA** and **REBA** ergonomic risk scores
- Derive **orientation data** (roll, pitch, yaw) from optical flow — **no hardware IMU required**
- Display everything on a **live Flask + Socket.IO web dashboard**
- **Data Collection**: Record multi-joint kinematic sessions to timestamped CSV files
- **Automated Reporting**: Generate professional **PDF ergonomic risk reports** with analytics and charts

The system runs on **NVIDIA Jetson Orin** (Ubuntu) or any Linux/Windows machine with Python 3.10+.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OAK-D Camera(s)                          │
│          RGB 1280×720 @ 30fps + Stereo Depth (aligned)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │  USB3 / XLink
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     camera/manager.py                           │
│  • Non-blocking queues (size=1, device + host side)             │
│  • tryGetAll() pattern → always freshest frame (no lag)         │
│  • Depth post-processing: speckle, temporal, spatial filters    │
└──────────┬──────────────────────┬───────────────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐    ┌─────────────────────┐
│  camera/         │    │  camera/             │
│  imu_manager.py  │    │  calibration.py      │
│  Visual IMU via  │    │  Intrinsics/extrinsics│
│  Optical Flow    │    │  from OAK-D          │
│  (LK sparse OF)  │    └─────────────────────┘
└──────────┬───────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      pose/estimator.py                          │
│         MediaPipe Pose → 33 body keypoints (x, y, z)            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       pose/fusion.py                            │
│   Multi-camera landmark fusion (average or triangulation)        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      pose/skeleton.py                           │
│  Joint angle computation: neck, trunk, upper/lower arm, wrist   │
└──────────┬────────────────────────┬────────────────────────────┘
           │                        │
           ▼                        ▼
┌─────────────────┐       ┌──────────────────┐
│ ergonomics/     │       │ ergonomics/       │
│ rula.py         │       │ reba.py           │
│ RULA 1–7 score  │       │ REBA 1–15 score   │
│ Group A+B table │       │ Group A+B table   │
└────────┬────────┘       └────────┬──────────┘
         └────────────┬────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  web/socket_events.py                           │
│   Background thread @ 10 Hz → emits pose_update via Socket.IO  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Web Dashboard (Flask)                       │
│   /dashboard  /camera  /rula  /reba  /3d                        │
│   Real-time charts, 3D skeleton, video feed, IMU telemetry      │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────┐    ┌──────────────────────┐
│ data/logger  │    │ reporting/            │
│ CSV session  │    │ PDF report generator  │
│ logging      │    │ + matplotlib graphs   │
└──────────────┘    └──────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| 🎥 **Multi-camera support** | 1 to 3 OAK-D cameras auto-detected and fused |
| 📐 **Depth-aligned RGB** | Stereo depth aligned to RGB frame; full post-processing pipeline |
| 🦴 **MediaPipe Pose** | 33 body keypoints, real-time, no GPU required |
| 📊 **RULA scoring** | Full 7-level risk score using exact official tables |
| 📊 **REBA scoring** | Full 15-level risk score using exact official tables |
| 🌀 **Visual IMU** | Roll/pitch/yaw from optical flow — no hardware IMU needed |
| 🌐 **Live web dashboard** | Flask + Socket.IO, accessible from any browser on the network |
| 🗄️ **Data logging** | CSV session logs with timestamps and all angle/score data |
| 📄 **PDF reports** | Professional risk reports with analytics, joint stats, and clinical recommendations |
| 🛡️ **Depth-Masking** | 0.5m – 3.0m depth filter to ignore background people and noise |
| ⚡ **Jetson Orin Ready** | Model complexity 0 + optimized queues for zero-lag performance |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Camera | 1× OAK-D Lite | 1–3× OAK-D (with BNO086 IMU) |
| Host CPU | 4-core ARM/x86 | NVIDIA Jetson Orin |
| RAM | 4 GB | 8 GB+ |
| USB | USB 3.0 | USB 3.1 Gen 2 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 / Windows 10+ |

> **Note:** The Visual IMU works on all OAK-D variants including **OAK-D Lite** (which has no physical IMU chip).  
> If your device has a **BNO086** IMU chip, you can re-enable hardware IMU in `camera/imu_manager.py`.

---

## Software Requirements

| Package | Version | Purpose |
|---|---|---|
| `depthai` | 2.24+ | OAK-D camera driver & pipeline |
| `mediapipe` | 0.10+ | Human pose estimation |
| `opencv-python` | 4.8+ | Image processing & optical flow |
| `flask` | 3.0+ | Web server |
| `flask-socketio` | 5.3+ | Real-time WebSocket communication |
| `numpy` | 1.24+ | Numerical computing |
| `pandas` | 2.0+ | Data logging & CSV export |
| `matplotlib` | 3.7+ | Chart generation for PDF reports |
| `reportlab` | 4.0+ | PDF report generation |
| `simple-websocket` | — | WebSocket backend for Flask-SocketIO |

---

## Project Structure

```
ergonomic-assessment-system/
│
├── app.py                      # Main entry point — orchestrates all modules
├── config.py                   # Global configuration (FPS, paths, thresholds)
├── requirements.txt            # Python dependencies
├── run.sh                      # Shell launcher script
├── rula_text.txt               # Official RULA scoring tables (reference)
├── reba_text.txt               # Official REBA scoring tables (reference)
│
├── camera/
│   ├── manager.py              # OAK-D RGB + Stereo Depth pipeline
│   ├── imu_manager.py          # Visual IMU (optical flow — no hardware IMU)
│   └── calibration.py         # Camera intrinsics / extrinsics
│
├── pose/
│   ├── estimator.py            # MediaPipe Pose → 33 landmarks
│   ├── fusion.py               # Multi-camera landmark fusion
│   └── skeleton.py             # Joint angle computation
│
├── ergonomics/
│   ├── rula.py                 # RULA calculator (Group A+B tables, score 1–7)
│   ├── reba.py                 # REBA calculator (Group A+B tables, score 1–15)
│   └── risk.py                 # Risk anomaly detector
│
├── web/
│   ├── routes.py               # Flask routes (/dashboard, /camera, /rula, /reba, /3d)
│   ├── socket_events.py        # Socket.IO event handlers + 10 Hz processing loop
│   ├── mock_server.py          # Mock server for testing without camera
│   ├── static/
│   │   ├── css/style.css       # Dashboard stylesheet (glassmorphic dark theme)
│   │   └── js/
│   │       ├── dashboard.js    # Main dashboard Socket.IO client + charts
│   │       └── 3d_viewer.js    # Three.js 3D skeleton viewer
│   └── templates/
│       ├── dashboard.html      # Main dashboard
│       ├── camera.html         # Live camera feed + depth view
│       ├── rula.html           # RULA detail page
│       ├── reba.html           # REBA detail page
│       └── 3d.html             # Interactive 3D skeleton view
│
├── data/
│   ├── logger.py               # Session data logger (CSV)
│   └── session_manager.py      # Session lifecycle management
│
├── reporting/
│   ├── report_generator.py     # PDF report builder
│   └── graphs.py               # Matplotlib chart helpers
│
└── examples/                   # Official DepthAI examples (reference only)
    ├── ColorCamera/
    ├── StereoDepth/
    ├── IMU/
    └── ...
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/charrada1993/ERGO-VISION.git
cd ERGO-VISION
```

2. **Initialize Environment**:
```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install DepthAI udev rules — Linux only

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 5. Connect your OAK-D camera via USB 3.0

---

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run the system
python3 app.py
```

Open your browser and navigate to:

```
http://localhost:5000/dashboard
```

### Available pages

| URL | Description |
|---|---|
| `/dashboard` | Main real-time dashboard with all metrics |
| `/camera` | Live RGB + depth video feed |
| `/rula` | RULA score breakdown (Group A, B, sub-scores) |
| `/reba` | REBA score breakdown (Group A, B, sub-scores) |
| `/3d` | Interactive Three.js 3D skeleton viewer |
| `/collection` | **[NEW]** Data collection page (Start/Stop recording to CSV) |
| `/report` | **[NEW]** Report generation engine (CSV → PDF) |

---

## Modules Documentation

### Camera Manager

**File:** `camera/manager.py`

Manages the OAK-D camera pipeline: RGB (1280×720 @ 30fps) and stereo depth (aligned to RGB).

**Key design decisions:**
- `xout.input.setBlocking(False)` + `xout.input.setQueueSize(1)` — eliminates device-side frame buildup
- `tryGetAll()[-1]` pattern on host side — always processes the freshest frame
- Depth post-processing: speckle filter, temporal filter, spatial filter (hole-filling), threshold filter (0.3–8 m)

```python
from camera.manager import CameraManager

cam = CameraManager(pipeline=pipeline, device=device)
cam.setup()          # Add nodes to pipeline (before device start)
cam.start_streams()  # Open queues and start background thread

frames = cam.get_latest_frames()
# Returns: {'rgb': np.ndarray, 'depth': np.ndarray, 'disp': np.ndarray, 'timestamp': float}
```

---

### Visual IMU

**File:** `camera/imu_manager.py`

Derives orientation (roll, pitch, yaw) and pseudo-motion data from RGB camera optical flow. **No hardware IMU chip required** — works with OAK-D Lite.

**Algorithm:**
1. **Lucas-Kanade sparse optical flow** tracks up to 200 feature points between consecutive frames
2. Mean flow vector → pseudo-accelerometer (ax, ay)
3. Homography estimated from matched points → in-plane rotation → yaw
4. Pitch and roll computed from the **exact reference formulas**:

```
Pitch (θ) = atan2(ax, √(ay² + az²)) × 180/π
Roll  (φ) = atan2(ay, √(ax² + az²)) × 180/π
Yaw   (ψ) = integrated from homography rotation angle
```

5. Exponential Moving Average (α=0.3) smooths all outputs

```python
from camera.imu_manager import IMUManager

imu = IMUManager()
imu.setup()
imu.set_camera_manager(cam_mgr)
imu.start()

data = imu.get_data()
# Returns: {'roll': float, 'pitch': float, 'yaw': float (degrees),
#           'accel': (x, y, z), 'gyro': (x, y, z), 'euler': (roll, pitch, yaw)}
```

---

### Pose Estimation

**File:** `pose/estimator.py`, `pose/fusion.py`, `pose/skeleton.py`

- **PoseEstimator**: wraps MediaPipe Pose, returns 33×3 landmark array (normalized x, y, z)
- **PoseFusion**: merges landmarks from multiple cameras (average or triangulation)
- **SkeletonBuilder**: computes joint angles from landmark positions

**Computed angles:**

| Angle Key | Description |
|---|---|
| `neck` | Neck flexion/extension (degrees) |
| `trunk` | Trunk forward lean (degrees) |
| `upper_arm_left/right` | Shoulder elevation (degrees) |
| `elbow_left/right` | Elbow angle (degrees) |
| `wrist_left/right` | Wrist flexion/deviation |

---

### Ergonomic Scoring (RULA & REBA)

**Files:** `ergonomics/rula.py`, `ergonomics/reba.py`

Both calculators implement the **exact official lookup tables** as documented in `rula_text.txt` and `reba_text.txt`.

```python
from ergonomics.rula import RULACalculator
from ergonomics.reba import REBACalculator

rula = RULACalculator()
reba = REBACalculator()

result = rula.compute(angles, load_kg=0, repetitive=False)
# result['RULA_score']     → 1–7
# result['risk_level']     → text label
# result['score_A'], ['score_B'], ['score_C']
# result['upper_arm_score'], ['lower_arm_score'], ['wrist_score'], ...

result = reba.compute(angles, load_kg=0, grip=0, repetitive=False)
# result['REBA_score']     → 1–15
# result['table_A'], ['table_B'], ['score_C']
# result['trunk_score'], ['neck_score'], ['legs_score'], ...
```

---

### Web Dashboard

**Files:** `web/routes.py`, `web/socket_events.py`

- **Flask** serves all HTML pages
- **Socket.IO** emits `pose_update` events at ~10 Hz with:
  - Joint angles
  - RULA/REBA scores and sub-scores
  - Risk level and anomaly list
  - Visual IMU data (roll, pitch, yaw)
- **Three.js** renders the 3D skeleton viewer at `/3d`

---

### Data Logging & Reports

**Files:** `data/logger.py`, `reporting/report_generator.py`

- **Data Collection (`/collection`)**:
  - Live stream with recording status indicators.
  - Generates timestamped CSV files in `data_sessions/`.
  - Logs: `timestamp`, `frame_id`, `joint_angles`, `RULA/REBA scores`, `anomalies`.
- **Report Generation (`/report`)**:
  - Automatically lists available recorded sessions.
  - Generates comprehensive PDF reports in `reports/` including:
    - **Executive Summary**: Mean/Peak risk levels.
    - **Joint Statistics**: Min/Max/Mean/95th percentile analysis.
    - **AI Insights**: Identification of critical joints and anomaly counts.
    - **Visual Analytics**: Time-series graphs for risk and posture.
    - **Clinical Recommendations**: Dynamic suggestions based on data.

---

## RULA Scoring Reference

### Group A — Upper Limb

| Element | Range | Score |
|---|---|---|
| **Upper arm** | Along body (0°) | 1 |
| | < 20° elevation | 2 |
| | 20°–45° | 2 |
| | 45°–90° | 3 |
| | > 90° | 4 |
| | +rotation / +abduction | +1 each |
| **Lower arm** | 60°–100° | 1 |
| | < 60° or > 100° | 2 |
| **Wrist** | 0°–15° | 1 |
| | 15°–30° | 2 |
| | > 30° | 3 |
| | +lateral deviation | +1 |
| **Load** | < 2 kg | 0 |
| | ≤ 10 kg / repetitive | 1 |
| | > 10 kg | 2 |

### Group B — Neck, Trunk, Legs

| Element | Range | Score |
|---|---|---|
| **Neck** | 0°–10° | 1 |
| | 10°–20° | 2 |
| | > 20° | 3 |
| | Extension | 4 |
| | +rotation / +lateral tilt | +1 each |
| **Trunk** | Upright (0°) | 1 |
| | 0°–20° | 2 |
| | 20°–60° | 3 |
| | > 60° | 4 |
| | Extension | 2 |
| | +rotation / +lateral tilt | +1 each |
| **Legs** | Stable | 1 |
| | Unstable | 2 |

### RULA Final Score Interpretation

| Score | Risk Level | Action |
|---|---|---|
| 1–2 | ✅ Acceptable | No action required |
| 3–4 | ⚠️ Low | Monitor |
| 5–6 | 🔶 Medium | Change needed |
| 7 | 🔴 Very High | Immediate action |

---

## REBA Scoring Reference

### Group A — Trunk, Neck, Legs

| Element | Range | Score |
|---|---|---|
| **Trunk** | Upright (0°) | 1 |
| | 0°–20° | 2 |
| | 20°–60° | 3 |
| | > 60° | 4 |
| | Extension | 2 |
| **Neck** | 0°–20° flexion | 1 |
| | > 20° / extension | 2 |
| **Legs** | Sitting / stable standing | 1 |
| | One knee bent / uneven weight | 2 |
| | Squatting / strongly bent knees | 3 |
| | Very unstable / moving | 4 |

### Group B — Upper Limb

| Element | Range | Score |
|---|---|---|
| **Upper arm** | ≤ 20° | 1 |
| | 20°–45° | 2 |
| | 45°–90° | 3 |
| | > 90° | 4 |
| **Lower arm** | 60°–100° | 1 |
| | < 60° or > 100° | 2 |
| **Wrist** | Neutral | 1 |
| | Flex/ext > 15° | 2 |
| | +lateral deviation | +1 |

### REBA Adjustments

| Factor | Score |
|---|---|
| Load < 5 kg | 0 |
| Load 5–10 kg | 1 |
| Load > 10 kg | 2 |
| Good grip | +0 |
| Average grip | +1 |
| Poor grip | +2 |
| Repetitive / prolonged posture | +1 |

### REBA Final Score Interpretation

| Score | Risk Level | Action |
|---|---|---|
| 1 | ✅ Negligible | No action |
| 2–3 | ⚠️ Low | Monitor |
| 4–7 | 🔶 Medium | Improvement needed |
| 8–10 | 🔴 High | Rapid intervention |
| 11–15 | 🆘 Very High | Immediate action |

---

## Visual IMU Formulas

The Visual IMU module derives orientation from optical flow using standard inertial navigation formulas:

```
Pitch (θ) = atan2(ax,  √(ay² + az²)) × 180/π
Roll  (φ) = atan2(ay,  √(ax² + az²)) × 180/π
Yaw   (ψ) = ∫ ω_z dt   (integrated from in-plane homography rotation)
```

Where `ax`, `ay` are derived from the mean optical flow displacement vector scaled to pseudo-acceleration units, and `az = 0` (2D camera has no Z-axis flow).

Smoothing is applied via **Exponential Moving Average** with α = 0.3:

```
value_smoothed = α × value_raw + (1 - α) × value_previous
```

---

## Configuration

Edit `config.py` to adjust system parameters:

```python
class Config:
    MAX_CAMERAS      = 3       # Maximum OAK-D cameras (1–3)
    PROCESSING_FPS   = 10      # Pose estimation frequency (Hz)
    LOG_INTERVAL     = 0.5     # CSV logging interval (seconds)

    # Ergonomic defaults
    LOAD_KG_DEFAULT     = 0    # Default carried load (kg)
    REPETITIVE_DEFAULT  = False
    PROLONGED_DEFAULT   = False
    GRIP_QUALITY_DEFAULT = 0   # 0=good, 1=average, 2=poor
```

Camera-specific tuning in `camera/manager.py`:

```python
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_720_P
FPS             = 30           # Camera frame rate
```

Visual IMU tuning in `camera/imu_manager.py`:

```python
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, ...)
_REDETECT_INTERVAL = 10        # Re-detect features every N frames
_EMA_ALPHA         = 0.3       # Smoothing factor (0=no change, 1=no smooth)
_SCALE_ACCEL       = 0.02      # Flow px/frame → pseudo m/s²
_SCALE_GYRO        = 0.5       # Rotation rad/frame → pseudo rad/s
```

---

## API Reference

### Socket.IO Events

#### Client → Server
| Event | Payload | Description |
|---|---|---|
| `connect` | — | Client connects; server responds with `config` |

#### Server → Client
| Event | Payload | Description |
|---|---|---|
| `config` | `{mode: int, usb3: bool}` | Number of cameras and USB speed |
| `pose_update` | See below | Real-time posture data at ~10 Hz |
| `skeleton_3d` | `{landmarks: [[x,y,z]×33]}` | 33 body landmarks for 3D viewer |

**`pose_update` payload:**
```json
{
  "angles":       {"neck": 12.3, "trunk": 5.1, "upper_arm_left": 45.2, ...},
  "rula":         4,
  "reba":         7,
  "risk_level":   "Moyen – Changement nécessaire",
  "anomalies":    ["Neck flexion: 42.0° (>40°)"],
  "rula_details": {"score_a": 3, "score_b": 4, "score_c": 4, ...},
  "reba_details": {"table_a": 5, "table_b": 6, "score_c": 7, ...},
  "imu": {
    "roll":  2.5,
    "pitch": -1.2,
    "yaw":   15.0,
    "accel": {"x": 0.001, "y": -0.002, "z": 0.0},
    "gyro":  {"x": 0.0, "y": 0.0, "z": 0.003}
  }
}
```

---

## Troubleshooting

### No OAK-D device found
```
[Main] No OAK-D device found. Exiting.
```
- Ensure the camera is connected via **USB 3.0** (not USB 2.0 — the cable matters)
- On Linux: install udev rules (see Installation step 4)
- Try `lsusb | grep 03e7` to confirm the device is detected by the OS

### Camera lag / stuttering
- Already fixed: the system uses `tryGetAll()[-1]` + `setQueueSize(1)` on both device and host
- If still laggy: reduce `FPS` in `manager.py` from 30 to 15

### Pipeline failed for device
```
[Main] Pipeline start failed for device XXXX: ...
```
- Disconnect and reconnect the camera
- Try a different USB port or cable
- Ensure no other application is using the camera

### Visual IMU shows zeros
- The IMU starts tracking after the first two frames; wait 1–2 seconds
- Ensure the camera is not completely static (no features to track)
- The IMU requires `set_camera_manager()` to be called before `start()`

### MediaPipe no landmarks detected
```
[Processing] No landmarks detected from any camera
```
- Ensure the subject is fully visible in the camera frame
- Improve lighting conditions
- Reduce `model_complexity` in `pose/estimator.py` from `1` to `0` for faster detection

### PDF report not generated
- Check that `reports/` directory is writable
- Ensure `reportlab` and `matplotlib` are installed: `pip install reportlab matplotlib`

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Code Style
- Follow PEP 8
- Add docstrings to all new methods
- Use the existing module structure (camera / pose / ergonomics / web / data)
- Reference official DepthAI examples in `examples/` for any camera-related changes

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Luxonis DepthAI](https://docs.luxonis.com/) — OAK-D camera SDK and examples
- [Google MediaPipe](https://developers.google.com/mediapipe) — Human pose estimation
- [McAtamney & Corlett (1993)](https://doi.org/10.1080/00140139308967424) — RULA methodology
- [Hignett & McAtamney (2000)](https://doi.org/10.1016/S0003-6870(99)00044-1) — REBA methodology

---

*Built with ❤️ for occupational health and safety.*
