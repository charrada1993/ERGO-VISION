# config.py
import os

class Config:
    # Camera
    MAX_CAMERAS = 1              # Single OAK-D camera only
    PROCESSING_FPS = 10          # Target pose estimation frequency (Hz)
    LOG_INTERVAL = 0.5           # Log data every 0.5 seconds

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
    # USB 2.0 constraint: keep RGB small to avoid saturating the bus.
    # 416×320 @ 8 fps = ~4 MB/s for RGB (comfortably within USB2 budget)
    CAMERA_FPS          = 8           # fps for RGB + mono cameras
    RGB_WIDTH           = 416         # pixels – USB2-safe resolution
    RGB_HEIGHT          = 320

    # 320P mono cameras (~213 KB/frame × 2 × 8 fps ≈ 3.4 MB/s stereo)
    MONO_RESOLUTION     = "THE_320_P"

    # StereoDepth post-processing – minimal to save ARM CPU cycles
    STEREO_SPECKLE_RANGE        = 50
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

    # ── Visual IMU (optical flow) ──────────────────────────────────────
    IMU_MAX_CORNERS     = 100         # was 200
    IMU_WIN_SIZE        = (15, 15)    # was (21, 21)
    IMU_REDETECT_INTERVAL = 20        # re-detect features every N frames
    IMU_HALF_RES        = True        # resize frame to half before flow

    # ── Socket.IO emit throttling ──────────────────────────────────────
    SKELETON_EMIT_EVERY  = 3          # emit skeleton_3d every N frames
    IMU_SAMPLE_EVERY     = 3          # re-sample IMU every N frames
    DEPTH_MASK_EVERY     = 2          # apply depth mask every N frames

    # ── MJPEG streaming ───────────────────────────────────────────────
    VIDEO_JPEG_QUALITY  = 55          # 0-100 (lower = faster encode)
    VIDEO_WIDTH         = 416
    VIDEO_HEIGHT        = 320
    DEPTH_JPEG_QUALITY  = 50
    DEPTH_WIDTH         = 320
    DEPTH_HEIGHT        = 180
    DEPTH_STREAM_FPS    = 5           # Hz – depth colormap is CPU-heavy