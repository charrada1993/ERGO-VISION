# config.py
import os

class Config:
    # Camera
    MAX_CAMERAS = 1              # Single OAK-D camera only
    PROCESSING_FPS = 8           # Target pose estimation frequency (Hz) — matches camera FPS
    LOG_INTERVAL = 1.0           # Log data every 1.0 second (halved from 0.5 to save I/O)

    # Paths (absolute or relative to project root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SESSION_DIR = os.path.join(BASE_DIR, "sessions")
    REPORT_DIR  = os.path.join(BASE_DIR, "reports")
    STATIC_DIR  = os.path.join(BASE_DIR, "web", "static")
    TEMPLATE_DIR = os.path.join(BASE_DIR, "web", "templates")

    # Ergonomic thresholds
    LOAD_KG_DEFAULT      = 0        # Default load if not entered by user
    REPETITIVE_DEFAULT   = False
    PROLONGED_DEFAULT    = False
    GRIP_QUALITY_DEFAULT = 0         # 0=good, 1=average, 2=poor

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.SESSION_DIR, exist_ok=True)
        os.makedirs(Config.REPORT_DIR,  exist_ok=True)
        os.makedirs(Config.STATIC_DIR,  exist_ok=True)
        os.makedirs(Config.TEMPLATE_DIR, exist_ok=True)


class JetsonConfig:
    """
    Performance tuning constants for NVIDIA Jetson Orin reComputer J3011.
    Connection: USB 2.0  |  RAM: 8 GB  |  SSD: 128 GB
    USB 2.0 theoretical bandwidth ≈ 480 Mbit/s (~60 MB/s real).
    Keep combined RGB + Depth throughput well below 40 MB/s to avoid
    packet loss and camera disconnects.
    """

    # ── OAK-D Camera pipeline ──────────────────────────────────────────
    # USB 2.0 constraint: only one OAK-D device can be used at a time.
    MAX_CAMERAS         = 1           # USB 2.0 cannot sustain more than one OAK-D

    # USB 2.0 constraint: keep RGB small to avoid saturating the bus.
    # 640×360 @ 8 fps = ~5.5 MB/s for RGB (comfortably within USB2 budget)
    CAMERA_FPS          = 8           # fps for RGB + mono cameras
    RGB_WIDTH           = 1280
    RGB_HEIGHT          = 720

    # THE_400_P (640×400) — lowest valid DepthAI mono resolution; THE_320_P does not exist
    MONO_RESOLUTION     = "THE_400_P"  # Lowest valid DepthAI mono res (640x400)

    # StereoDepth post-processing – minimal to save ARM CPU cycles
    STEREO_SPECKLE_RANGE        = 28          # was 50; reduced to fit OAK-D memory
    STEREO_SPATIAL_RADIUS       = 2
    STEREO_SPATIAL_ITERATIONS   = 1
    STEREO_TEMPORAL_ENABLE      = False   # Disabled: CPU-intensive on ARM
    STEREO_DEPTH_MIN_MM         = 300
    STEREO_DEPTH_MAX_MM         = 5000    # 5 m limit (was 8 m) – saves lookups

    # ── MediaPipe pose estimation ──────────────────────────────────────
    # Resize BEFORE MediaPipe – biggest single CPU saving.
    # Landmarks are normalised 0-1 so they map back to any resolution.
    POSE_INPUT_WIDTH    = 320
    POSE_INPUT_HEIGHT   = 180

    # ── Legacy IMU settings (for module integrity only) ────────────────
    IMU_MAX_CORNERS     = 100
    IMU_WIN_SIZE        = (15, 15)
    IMU_REDETECT_INTERVAL = 20
    IMU_HALF_RES        = True
    IMU_SAMPLE_EVERY     = 8

    # ── MJPEG Stream settings (Jetson optimized) ──────────────────────
    VIDEO_WIDTH         = 480         # 16:9 dashboard stream (was 640)
    VIDEO_HEIGHT        = 270
    VIDEO_JPEG_QUALITY  = 40          # 0-100 (was 55; lower = faster encode on ARM)
    VIDEO_STREAM_FPS    = 8           # explicit FPS cap for MJPEG generator
    
    DEPTH_WIDTH         = 320         # 16:9 depth stream
    DEPTH_HEIGHT        = 180
    DEPTH_JPEG_QUALITY  = 40          # depth map quality (was 50)
    DEPTH_STREAM_FPS    = 4           # Hz – depth colormap is CPU-heavy (was 8)

    # ── Socket.IO emit throttling ──────────────────────────────────────
    SKELETON_EMIT_EVERY  = 2          # emit skeleton_3d every N frames (4 Hz)
    POSE_UPDATE_MAX_HZ   = 4          # hard cap on pose_update events (Hz)
    DEPTH_MASK_EVERY     = 1          # apply depth mask every N frames (stable landmarks)

    # ── Process loop timing ───────────────────────────────────────────
    PROCESS_LOOP_SLEEP   = 0.005      # yield GIL after successful emit (5 ms)