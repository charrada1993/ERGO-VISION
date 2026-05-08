#!/bin/bash
# jetson_setup.sh — One-shot environment bootstrap for Jetson Orin reComputer J3011
# Run this ONCE after flashing JetPack to prepare the system for ERGO-VISION.
# USB 2.0 | 8 GB RAM | 128 GB SSD
#
# Usage:
#   chmod +x jetson_setup.sh
#   sudo ./jetson_setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================================"
echo "  ERGO-VISION — Jetson Orin Setup (USB 2.0 / 8 GB RAM)"
echo "============================================================"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────
echo "[Setup] Installing system packages …"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    curl \
    git

# ── 2. Swap configuration ─────────────────────────────────────────────────
# 4 GB swap is recommended for 8 GB RAM Jetson systems running ML workloads.
SWAP_FILE="/swapfile"
SWAP_SIZE_MB=4096

CURRENT_SWAP=$(free -m | awk '/^Swap:/{print $2}')
if [ "$CURRENT_SWAP" -lt 3500 ]; then
    echo "[Setup] Configuring ${SWAP_SIZE_MB} MB swap file at ${SWAP_FILE} …"
    if [ -f "$SWAP_FILE" ]; then
        swapoff "$SWAP_FILE" 2>/dev/null || true
        rm -f "$SWAP_FILE"
    fi
    fallocate -l "${SWAP_SIZE_MB}M" "$SWAP_FILE"
    chmod 600 "$SWAP_FILE"
    mkswap "$SWAP_FILE"
    swapon "$SWAP_FILE"
    # Persist across reboots
    if ! grep -q "$SWAP_FILE" /etc/fstab; then
        echo "${SWAP_FILE} none swap sw 0 0" >> /etc/fstab
    fi
    echo "[Setup] Swap configured: $(free -h | awk '/^Swap:/{print $2}')"
else
    echo "[Setup] Swap already sufficient: ${CURRENT_SWAP} MB — skipping"
fi

# ── 3. DepthAI / Luxonis OAK-D USB rules ──────────────────────────────────
# Install udev rules so OAK-D is accessible over USB 2.0 without root.
echo "[Setup] Installing DepthAI USB udev rules …"
if [ ! -f /etc/udev/rules.d/80-movidius.rules ]; then
    curl -fsSL \
        "https://raw.githubusercontent.com/luxonis/depthai/main/install_requirements.sh" \
        -o /tmp/dai_install.sh
    bash /tmp/dai_install.sh || echo "[Setup] DepthAI install script ran with warnings"
else
    echo "[Setup] DepthAI udev rules already installed — skipping"
fi

# ── 4. Python virtual environment ─────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/venv"
echo "[Setup] Creating Python venv at ${VENV_DIR} …"
python3 -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel

# Install Python dependencies (ARM64-safe)
echo "[Setup] Installing Python packages …"
pip install \
    depthai \
    mediapipe \
    flask \
    flask-socketio \
    simple-websocket \
    pandas \
    numpy \
    opencv-python-headless \
    reportlab \
    matplotlib

deactivate

# ── 5. Jetson power mode ───────────────────────────────────────────────────
echo "[Setup] Setting Jetson to MAXN performance mode …"
nvpmodel -m 0  2>/dev/null || echo "[Setup] nvpmodel not found (skip)"
jetson_clocks  2>/dev/null || echo "[Setup] jetson_clocks not found (skip)"

# ── 6. Make scripts executable ────────────────────────────────────────────
chmod +x "${SCRIPT_DIR}/run.sh"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  To start the application:"
echo "    cd ${SCRIPT_DIR} && bash run.sh"
echo "  Web dashboard: http://<jetson-ip>:5000"
echo "============================================================"
echo ""
