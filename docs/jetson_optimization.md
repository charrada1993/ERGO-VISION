# Jetson Orin Optimization: Hardware Tuning Guide

*Last updated: 2026-05-11*

The ERGO-VISION system is tuned for the **NVIDIA Jetson Orin Nano (reComputer J3011, 8 GB)**. To achieve real-time performance (pose estimation + AI inference + video streaming simultaneously), several hardware and software optimizations are applied.

---

## 1. System Environment

| Component | Details |
|---|---|
| **Device** | NVIDIA Jetson Orin Nano (reComputer J3011) |
| **RAM** | 8 GB LPDDR5 |
| **OS** | Ubuntu 22.04 (JetPack 6.x) |
| **Python** | 3.10 (`oak_env` virtual environment) |
| **Camera** | OAK-D (USB 3.1 Gen 2) |

---

## 2. Power & Clock Management

```bash
sudo nvpmodel -m 0   # 15W max performance mode
sudo jetson_clocks   # Pin CPU/GPU to max frequency
```

| Setting | Effect |
|---|---|
| **NVPModel 0** | All CPU cores active, max frequency |
| **Jetson Clocks** | Eliminates Dynamic Frequency Scaling throttle spikes |

---

## 3. Memory Efficiency

```bash
export MALLOC_ARENA_MAX=2
```

| Optimization | Details |
|---|---|
| `MALLOC_ARENA_MAX=2` | Prevents GLIBC memory arena fragmentation on ARM |
| ZRAM 4 GB swap | Handles MediaPipe RAM spikes gracefully |
| NumPy-only AI | ErgoNet v2.0 uses ~15 MB RAM vs. ~2 GB for PyTorch |

---

## 4. AI Inference Speed

ErgoNet v2.0 achieves **< 8 ms** inference latency via:

- **OpenBLAS** — ARM v8.2 NEON vectorized matrix multiply
- **Single hidden layer (512)** — minimal matrix sizes
- **Pre-loaded normalization stats** — zero per-frame overhead at inference
- **No backpropagation** — operational model does forward pass only

Full inference pipeline per frame:
```
Angles → Z-score normalize → Forward pass → De-normalize → Socket.IO emit  (~8–12 ms total)
```

---

## 5. Camera Pipeline Settings

| Setting | Value | Reason |
|---|---|---|
| `setBlocking(False)` + `setQueueSize(1)` | Non-blocking, size 1 | Prevents pipeline stall; discards stale frames |
| `tryGetAll()[-1]` | Host side | Always processes the freshest frame |
| RGB resolution | 1280×720 @ 30 FPS | Balances accuracy and USB bandwidth |

---

## 6. Running the System

```bash
source ~/oak_env/bin/activate
cd ~/ERGO-VISION
python3 app.py
```

For CPU core isolation (optional):
```bash
taskset -c 0-3 python3 app.py
```

---

*Documented by ErgoVision AI Team · 2026*
