# Jetson Orin Optimization: Hardware Tuning

The ERGO-VISION system is specifically tuned for the **NVIDIA Jetson Orin Nano (reComputer J3011)**. To achieve real-time 8 FPS performance with AI tracking, several hardware-level optimizations were applied.

## 1. Power & Clock Management
The `run.sh` script applies the following to maximize the 8GB RAM and ARM cores:
- **NVPModel 15W**: Puts the Jetson into its high-performance power mode.
- **Jetson Clocks**: Pinned the CPU and GPU to their maximum frequencies (`sudo jetson_clocks`) to eliminate "Dynamic Frequency Scaling" lag.

## 2. Memory (RAM) Efficiency
With 8GB of RAM, we must be careful with memory fragmentation.
- **MALLOC_ARENA_MAX=2**: This environment variable prevents the GLIBC allocator from creating too many arenas, which can lead to memory exhaustion on ARM devices.
- **ZRAM Configuration**: The system uses a 4GB Swap-on-RAM (ZRAM) to handle sudden spikes in MediaPipe memory usage.

## 3. Computation Speed
- **Numpy BLAS**: Our custom AI uses OpenBLAS, which is highly optimized for the ARM v8.2 NEON instructions on the Orin. This allows the 100-output model to run in less than **0.5 milliseconds**.
- **Process Binding**: The main processing loop is bound to the first 4 CPU cores using `taskset -c 0-3`, ensuring it doesn't compete with background OS tasks.
