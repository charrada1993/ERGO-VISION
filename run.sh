#!/bin/bash
# run.sh — Launch ERGO-VISION on Jetson Orin reComputer J3011
# USB 2.0 mode | 8 GB RAM | Jetson Linux (JetPack 5.x or 6.x)
# Optimized: CPU clock pinning, GIL-friendly malloc, core affinity

set -e
cd "$(dirname "$0")"

echo "=== ERGO-VISION Jetson Orin Startup ==="

# ── 1. Jetson Performance Mode ────────────────────────────────────────────
# MAXN = all CPU cores + GPU at maximum frequency.
# This prevents thermal throttling during sustained pose estimation.
echo "[Jetson] Setting MAXN performance mode …"
sudo nvpmodel -m 0 2>/dev/null || echo "[Jetson] nvpmodel not available (skip)"

# jetson_clocks pins all CPU/GPU clocks to max — critical to avoid mid-session throttle.
# We attempt passwordless sudo first, then prompt if needed.
if sudo -n jetson_clocks 2>/dev/null; then
    echo "[Jetson] jetson_clocks applied (max clocks pinned)"
else
    echo "[Jetson] NOTE: Run 'sudo jetson_clocks' manually to pin CPU/GPU clocks"
fi

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
else
    echo "[Jetson] Swap OK: ${SWAP_MB} MB"
fi

# ── 4. Python environment ─────────────────────────────────────────────────
echo "[Jetson] Using system Python with --user packages"

# ── 5. Performance environment variables ─────────────────────────────────
# Reduce glibc memory arena fragmentation (cuts ~100 MB RAM waste on ARM).
export MALLOC_ARENA_MAX=2

# Help NumPy/OpenBLAS not over-thread on 6-core ARM (we manage threads manually).
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Disable Python hash randomization for reproducible runs.
export PYTHONHASHSEED=0

# ── 6. Log rotation (prevent backend.log from growing unbounded) ─────────────
LOG_FILE="backend.log"
MAX_LOG_MB=50
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE_MB=$(du -m "$LOG_FILE" | cut -f1)
    if [ "$LOG_SIZE_MB" -ge "$MAX_LOG_MB" ]; then
        echo "[Jetson] Log rotation: ${LOG_FILE} is ${LOG_SIZE_MB} MB — archiving"
        mv "$LOG_FILE" "${LOG_FILE%.log}_$(date +%Y%m%d_%H%M%S).log.bak"
        # Keep only 2 backups
        ls -t backend_*.log.bak 2>/dev/null | tail -n +3 | xargs rm -f 2>/dev/null || true
    fi
fi

# ── 7. Launch application ─────────────────────────────────────────────────
echo "[Jetson] Starting ERGO-VISION … (http://0.0.0.0:5000)"
echo "[Jetson] Resources: $(free -h | awk '/^Mem:/{print $3"/"$2" RAM used"}')"

# taskset: bind Python to CPUs 0-3 (big Cortex-A78 cores on Orin Nano).
# This keeps the OS and USB driver on CPUs 4-5 and avoids cache thrashing.
if command -v taskset &>/dev/null; then
    exec taskset -c 0-3 python3 app.py 2>&1 | tee -a "$LOG_FILE"
else
    exec python3 app.py 2>&1 | tee -a "$LOG_FILE"
fi