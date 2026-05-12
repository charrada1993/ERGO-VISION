#!/usr/bin/env python3
"""
system_test.py — ERGO-VISION Pre-flight Health Checker
Jetson Orin reComputer J3011 | 8 GB RAM | 128 GB SSD

Run this BEFORE starting the application to verify all components:
    python3 system_test.py

Exit code: 0 = all PASS/WARN, 1 = at least one FAIL
"""

import sys
import os
import time
import subprocess
import platform

# ── ANSI colours ──────────────────────────────────────────────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"

PASS = f"{GRN}{BLD}[ PASS ]{RST}"
WARN = f"{YLW}{BLD}[ WARN ]{RST}"
FAIL = f"{RED}{BLD}[ FAIL ]{RST}"
INFO = f"{CYN}{BLD}[ INFO ]{RST}"

results = []   # list of (status, message)

def check(label, status, detail=""):
    tag = PASS if status == "pass" else (WARN if status == "warn" else FAIL)
    line = f"  {tag}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    results.append(status)


# ─────────────────────────────────────────────────────────────────────────────
print()
print(f"  {BLD}{'='*58}{RST}")
print(f"  {BLD}  ERGO-VISION — Jetson Orin Pre-flight System Test{RST}")
print(f"  {BLD}{'='*58}{RST}")
print()

# ── 1. Platform ───────────────────────────────────────────────────────────────
print(f"  {CYN}▶  1. Platform{RST}")
arch  = platform.machine()
kver  = platform.release()
pyver = sys.version.split()[0]
check("Architecture",  "pass" if arch == "aarch64" else "warn", arch)
check("Kernel",        "pass", kver)
check("Python version","pass" if pyver.startswith("3.10") else "warn", pyver)
print()

# ── 2. RAM & Swap ─────────────────────────────────────────────────────────────
print(f"  {CYN}▶  2. Memory{RST}")
try:
    mem_info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            mem_info[k.strip()] = int(v.strip().split()[0])   # kB

    total_mb  = mem_info["MemTotal"]  // 1024
    avail_mb  = mem_info["MemAvailable"] // 1024
    swap_mb   = mem_info.get("SwapTotal", 0) // 1024

    check("Total RAM",  "pass" if total_mb >= 7000 else "warn",
          f"{total_mb} MB")
    check("Available RAM", "pass" if avail_mb >= 2000 else "warn",
          f"{avail_mb} MB  (need ≥2 GB free)")
    check("Swap",       "pass" if swap_mb >= 3000 else "warn",
          f"{swap_mb} MB  (recommend ≥4 GB)")
except Exception as e:
    check("Memory read", "fail", str(e))
print()

# ── 3. SSD / Disk ─────────────────────────────────────────────────────────────
print(f"  {CYN}▶  3. Storage{RST}")
try:
    st = os.statvfs("/")
    free_gb = (st.f_bavail * st.f_frsize) // (1024**3)
    total_gb = (st.f_blocks * st.f_frsize) // (1024**3)
    check("Disk free", "pass" if free_gb >= 10 else "warn",
          f"{free_gb}/{total_gb} GB")
except Exception as e:
    check("Disk check", "fail", str(e))
print()

# ── 4. Jetson Power Mode ──────────────────────────────────────────────────────
print(f"  {CYN}▶  4. Jetson Performance{RST}")
try:
    r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
    mode_line = [l for l in r.stdout.splitlines() if "Power Mode" in l]
    mode_str  = mode_line[0] if mode_line else r.stdout.strip()
    is_maxn   = "0" in r.stdout or "MAXN" in r.stdout.upper()
    check("nvpmodel", "pass" if is_maxn else "warn", mode_str)
except FileNotFoundError:
    check("nvpmodel", "warn", "not found — skip")
except Exception as e:
    check("nvpmodel", "warn", str(e))

try:
    r2 = subprocess.run(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"],
                        capture_output=True, text=True, timeout=3)
    freq_mhz = int(r2.stdout.strip()) // 1000
    check("CPU0 frequency", "pass" if freq_mhz >= 1400 else "warn",
          f"{freq_mhz} MHz  (run 'sudo jetson_clocks' to pin to max)")
except Exception:
    check("CPU frequency", "warn", "could not read")

try:
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        temp_c = int(f.read().strip()) / 1000
    check("CPU temperature", "pass" if temp_c < 70 else "warn",
          f"{temp_c:.1f} °C")
except Exception:
    check("CPU temperature", "warn", "could not read")
print()

# ── 5. Python Imports ─────────────────────────────────────────────────────────
print(f"  {CYN}▶  5. Python Dependencies{RST}")
deps = [
    ("numpy",          "numpy"),
    ("opencv-python",  "cv2"),
    ("flask",          "flask"),
    ("flask-socketio", "flask_socketio"),
    ("simple-websocket","simple_websocket"),
    ("mediapipe",      "mediapipe"),
    ("depthai",        "depthai"),
    ("pandas",         "pandas"),
    ("reportlab",      "reportlab"),
    ("matplotlib",     "matplotlib"),
]
for pkg_name, mod_name in deps:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "?")
        check(pkg_name, "pass", ver)
    except ImportError as e:
        check(pkg_name, "fail", f"NOT INSTALLED — run sudo bash jetson_setup.sh")
print()

# ── 6. ERGO-VISION Module Imports ─────────────────────────────────────────────
print(f"  {CYN}▶  6. ERGO-VISION Internal Modules{RST}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
internal = [
    ("config.Config",           "config",               "Config"),
    ("config.JetsonConfig",     "config",               "JetsonConfig"),
    ("camera.manager",          "camera.manager",       "CameraManager"),
    ("camera.imu_manager",      "camera.imu_manager",   "IMUManager"),
    ("pose.estimator",          "pose.estimator",       "PoseEstimator"),
    ("pose.fusion",             "pose.fusion",          "PoseFusion"),
    ("pose.skeleton",           "pose.skeleton",        "SkeletonBuilder"),
    ("ergonomics.rula",         "ergonomics.rula",      "RULACalculator"),
    ("ergonomics.reba",         "ergonomics.reba",      "REBACalculator"),
    ("data.logger",             "data.logger",          "DataLogger"),
    ("web.routes",              "web.routes",           "create_app"),
    ("web.socket_events",       "web.socket_events",    "SocketEvents"),
]
for label, mod_path, attr in internal:
    try:
        mod = __import__(mod_path, fromlist=[attr])
        getattr(mod, attr)
        check(label, "pass")
    except Exception as e:
        check(label, "fail", str(e)[:80])
print()

# ── 7. MediaPipe Smoke Test ───────────────────────────────────────────────────
print(f"  {CYN}▶  7. MediaPipe Pose Smoke Test{RST}")
try:
    import numpy as np
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    t0 = time.time()
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as pose:
        # Synthetic 320×180 BGR frame (gradient, not all-black)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        frame[:, :, 1] = np.linspace(0, 128, 320, dtype=np.uint8)  # green gradient
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose.process(rgb)   # may or may not detect a person — that's fine
    elapsed_ms = (time.time() - t0) * 1000
    check("MediaPipe init+process", "pass" if elapsed_ms < 3000 else "warn",
          f"{elapsed_ms:.0f} ms (first-run JIT compile)")
except Exception as e:
    check("MediaPipe smoke test", "fail", str(e)[:80])
print()

# ── 8. OAK-D Camera ───────────────────────────────────────────────────────────
print(f"  {CYN}▶  8. OAK-D Camera{RST}")
try:
    import depthai as dai
    devs = dai.Device.getAllAvailableDevices()
    if devs:
        for d in devs:
            dev_id = getattr(d, 'getMxId', getattr(d, 'getDeviceId', lambda: getattr(d, 'name', 'Unknown')))()
            check(f"OAK-D device", "pass",
                  f"MxID={dev_id}  USB={d.state.name}")
    else:
        check("OAK-D device", "warn",
              "No camera found — connect OAK-D via USB then re-run")
except Exception as e:
    check("OAK-D scan", "fail", str(e)[:80])
print()

# ── 9. Web Templates ──────────────────────────────────────────────────────────
print(f"  {CYN}▶  9. Web Templates & Static{RST}")
base = os.path.dirname(os.path.abspath(__file__))
required_templates = [
    "dashboard.html", "camera.html",
    "rula.html", "reba.html",
    "3d.html", "collection.html", "report.html",
    "ai.html",
]
tmpl_dir = os.path.join(base, "web", "templates")
for t in required_templates:
    path = os.path.join(tmpl_dir, t)
    check(f"template/{t}", "pass" if os.path.exists(path) else "warn",
          "" if os.path.exists(path) else "MISSING")
print()

# ── 10. AI Model ─────────────────────────────────────────────────────────────
print(f"  {CYN}▶  10. AI Model{RST}")
models_dir = os.path.join(base, "ai", "models")
v2_path = os.path.join(models_dir, "ergo_net_v2.pkl")
v1_path = os.path.join(models_dir, "ergo_net_numpy.pkl")
if os.path.exists(v2_path):
    size_mb = os.path.getsize(v2_path) / (1024 * 1024)
    check("ErgoNet v2.0 model", "pass", f"ergo_net_v2.pkl ({size_mb:.1f} MB)")
elif os.path.exists(v1_path):
    size_mb = os.path.getsize(v1_path) / (1024 * 1024)
    check("ErgoNet v1.0 model", "warn", f"ergo_net_numpy.pkl ({size_mb:.1f} MB) — run train_v2.py")
else:
    check("ErgoNet model", "fail",
          "No model found in ai/models/ — run: cd ai && python3 train_v2.py")

try:
    from ai.operation.inference import NumpyInference
    inf = NumpyInference()
    dummy = {col: 10.0 for col in inf.input_cols}
    result = inf.predict(dummy)
    risk = result.get('risk_score', 'N/A')
    check("Inference smoke test", "pass",
          f"v{inf.version}: risk_score={risk:.2f}, outputs={list(result.keys())}")
except Exception as e:
    check("Inference smoke test", "fail", str(e)[:80])
print()

# ── Summary ───────────────────────────────────────────────────────────────────
n_pass = results.count("pass")
n_warn = results.count("warn")
n_fail = results.count("fail")

print(f"  {BLD}{'─'*58}{RST}")
print(f"  Results: {GRN}{BLD}{n_pass} PASS{RST}  "
      f"{YLW}{BLD}{n_warn} WARN{RST}  "
      f"{RED}{BLD}{n_fail} FAIL{RST}")
print(f"  {BLD}{'─'*58}{RST}")

if n_fail > 0:
    print(f"\n  {RED}{BLD}⚠  Fix FAIL items above, then re-run this test.{RST}")
    print(f"  {BLD}  Install missing deps:  sudo bash jetson_setup.sh{RST}\n")
    sys.exit(1)
elif n_warn > 0:
    print(f"\n  {YLW}{BLD}⚡  WARN items are non-fatal but may affect performance.{RST}\n")
    sys.exit(0)
else:
    print(f"\n  {GRN}{BLD}✅  All checks passed! Run:  bash run.sh{RST}\n")
    sys.exit(0)
