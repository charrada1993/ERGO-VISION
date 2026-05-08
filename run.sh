#!/bin/bash
# run.sh — Launch ERGO-VISION on Jetson Orin reComputer J3011
# USB 2.0 mode | 8 GB RAM | Jetson Linux (JetPack 5.x or 6.x)

set -e
cd "$(dirname "$0")"

echo "=== ERGO-VISION Jetson Orin Startup ==="

# ── 1. Jetson Performance Mode ────────────────────────────────────────────
# MAXN = all CPU cores + GPU at maximum frequency.
# This prevents thermal throttling during sustained pose estimation.
echo "[Jetson] Setting MAXN performance mode …"
sudo nvpmodel -m 0 2>/dev/null || echo "[Jetson] nvpmodel not available (skip)"
sudo jetson_clocks  2>/dev/null || echo "[Jetson] jetson_clocks not available (skip)"

# ── 2. USB power management ───────────────────────────────────────────────
# USB 2.0 OAK-D: prevent the OS from suspending the USB port mid-session.
echo "[Jetson] Disabling USB power management …"
for dev in /sys/bus/usb/devices/*/power/control; do
    echo 'on' | sudo tee "$dev" > /dev/null 2>&1 || true
done

# ── 3. Swap check (recommended: 4 GB for 8 GB RAM systems) ───────────────
SWAP_MB=$(free -m | awk '/^Swap:/{print $2}')
if [ "$SWAP_MB" -lt 2000 ]; then
    echo "[Jetson] WARNING: Only ${SWAP_MB} MB swap detected."
    echo "[Jetson] Run jetson_setup.sh to configure 4 GB swap."
fi

# ── 4. Python environment ─────────────────────────────────────────────────
# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[Jetson] Virtual environment activated"
fi

# ── 5. Launch application ─────────────────────────────────────────────────
echo "[Jetson] Starting ERGO-VISION … (http://0.0.0.0:5000)"
python3 app.py