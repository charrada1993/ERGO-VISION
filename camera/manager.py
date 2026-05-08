# camera/manager.py
# Optimized for NVIDIA Jetson Orin reComputer J3011 over USB 2.0.
# USB 2.0 budget: ~60 MB/s real. We target <10 MB/s total for RGB+Depth.
# Ref: depthai examples/StereoDepth/rgb_depth_aligned.py

import depthai as dai
import threading
import time
import numpy as np
import cv2
from config import JetsonConfig

# ── Resolved tuning from JetsonConfig ─────────────────────────────────────────
FPS             = JetsonConfig.CAMERA_FPS          # 8 fps – USB2 safe
RGB_SOCKET      = dai.CameraBoardSocket.CAM_A

# Mono resolution: THE_400_P (640×400) — lowest valid DepthAI mono resolution
# NOTE: THE_320_P does not exist in the DepthAI SensorResolution enum.
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P


class CameraManager:
    """
    Single-camera manager for one OAK-D device.
    Streams RGB + aligned StereoDepth at USB2-safe rates.
    All queues are non-blocking (size=1) to always serve the freshest frame.
    """

    def __init__(self, pipeline=None, device=None):
        self.pipeline    = pipeline    # Shared pipeline from app.py
        self.device      = device      # Shared device (set after pipeline start)
        self.running     = False

        # Latest frames – written by background thread, read by callers
        self.frame_rgb   = None        # BGR numpy array from RGB camera
        self.frame_depth = None        # uint16 depth in mm (aligned to RGB)
        self.frame_disp  = None        # raw disparity uint8 for visualisation
        self._lock       = threading.Lock()

        # StereoDepth node handle – needed at runtime to read maxDisparity
        self._stereo     = None
        self._max_disp   = None

    # ------------------------------------------------------------------
    # Pipeline setup (called BEFORE device is started)
    # ------------------------------------------------------------------
    def setup(self):
        """
        Configure the DepthAI pipeline for USB 2.0 operation.
        Uses conservative resolutions and disables heavy post-processing
        to stay within ARM CPU and USB bandwidth budgets.
        """
        if self.pipeline is None:
            print("[Camera] ERROR – no pipeline provided")
            return False

        # ── RGB camera ──────────────────────────────────────────────────
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(RGB_SOCKET)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # Scale 1080p down by 1/3 -> 640x360 (preserves 16:9 FOV, NO zoom/crop)
        cam_rgb.setIspScale(1, 3)
        cam_rgb.setVideoSize(JetsonConfig.RGB_WIDTH, JetsonConfig.RGB_HEIGHT)
        cam_rgb.setFps(FPS)

        # ── Mono cameras (left / right) ─────────────────────────────────
        mono_left  = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(MONO_RESOLUTION)
        mono_left.setCamera("left")
        mono_left.setFps(FPS)
        mono_right.setResolution(MONO_RESOLUTION)
        mono_right.setCamera("right")
        mono_right.setFps(FPS)

        # ── StereoDepth ─────────────────────────────────────────────────
        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

        # LR-check required for depth alignment
        stereo.setLeftRightCheck(True)

        # Align depth map to the RGB camera coordinate space
        stereo.setDepthAlign(RGB_SOCKET)

        # Lightweight 3×3 median (7×7 is too CPU-heavy on ARM)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)

        # Minimal post-processing – only speckle + spatial, NO temporal filter
        # (temporalFilter is the most CPU-intensive post-process on ARM)
        cfg = stereo.initialConfig.get()
        cfg.postProcessing.speckleFilter.enable       = True
        cfg.postProcessing.speckleFilter.speckleRange = JetsonConfig.STEREO_SPECKLE_RANGE

        cfg.postProcessing.temporalFilter.enable      = False  # DISABLED – saves ARM CPU

        cfg.postProcessing.spatialFilter.enable            = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = JetsonConfig.STEREO_SPATIAL_RADIUS
        cfg.postProcessing.spatialFilter.numIterations     = JetsonConfig.STEREO_SPATIAL_ITERATIONS

        cfg.postProcessing.thresholdFilter.minRange = JetsonConfig.STEREO_DEPTH_MIN_MM
        cfg.postProcessing.thresholdFilter.maxRange = JetsonConfig.STEREO_DEPTH_MAX_MM
        cfg.postProcessing.decimationFilter.decimationFactor = 1
        stereo.initialConfig.set(cfg)

        self._stereo = stereo  # keep reference for maxDisparity readback

        # ── XLink outputs ────────────────────────────────────────────────
        # Non-blocking, size=1: device drops old frames before USB transfer,
        # preventing any queue build-up over USB 2.0.
        xout_rgb   = self.pipeline.create(dai.node.XLinkOut)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_disp  = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        xout_disp.setStreamName("disp")

        xout_rgb.input.setBlocking(False)
        xout_rgb.input.setQueueSize(1)
        xout_depth.input.setBlocking(False)
        xout_depth.input.setQueueSize(1)
        xout_disp.input.setBlocking(False)
        xout_disp.input.setQueueSize(1)

        # ── Node linking ─────────────────────────────────────────────────
        cam_rgb.video.link(xout_rgb.input)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(xout_depth.input)       # raw 16-bit depth in mm
        stereo.disparity.link(xout_disp.input)    # raw disparity for visualisation

        print(
            f"[Camera] Pipeline configured: RGB {JetsonConfig.RGB_WIDTH}×"
            f"{JetsonConfig.RGB_HEIGHT} + StereoDepth 320P @ {FPS} fps | USB 2.0 mode"
        )
        return True

    # ------------------------------------------------------------------
    # Streaming (called AFTER device is started)
    # ------------------------------------------------------------------
    def start_streams(self):
        """Open output queues and launch the background reader thread."""
        if self.device is None:
            print("[Camera] ERROR – no device provided")
            return

        # Cache maxDisparity for normalisation
        if self._stereo is not None:
            try:
                self._max_disp = self._stereo.initialConfig.getMaxDisparity()
            except Exception:
                self._max_disp = 95.0  # safe default

        # Non-blocking host-side queues – always return the freshest frame
        self._q_rgb   = self.device.getOutputQueue("rgb",   maxSize=1, blocking=False)
        self._q_depth = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
        self._q_disp  = self.device.getOutputQueue("disp",  maxSize=1, blocking=False)

        self.running = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        print(f"[Camera] Streaming: RGB + Depth @ {FPS} fps (USB 2.0)")

    # ------------------------------------------------------------------
    # Background reader thread
    # ------------------------------------------------------------------
    def _reader(self):
        """
        Non-blocking frame drain loop.
        Uses tryGetAll() so we never block waiting for a frame – critical
        over USB 2.0 where latency spikes are common.
        Poll at 20 ms (50 Hz) so a new frame (arriving at 8 Hz = every 125 ms)
        is picked up within 20 ms of arrival instead of up to 125 ms.
        """
        while self.running:
            try:
                # 1. RGB
                packets = self._q_rgb.tryGetAll()
                if packets:
                    pkt = packets[-1]
                    with self._lock:
                        self.frame_rgb = pkt.getCvFrame()   # BGR uint8

                # 2. Depth
                packets = self._q_depth.tryGetAll()
                if packets:
                    pkt = packets[-1]
                    with self._lock:
                        self.frame_depth = pkt.getFrame()   # uint16 mm

                # 3. Disparity – store raw; colourise on-demand in routes.py
                packets = self._q_disp.tryGetAll()
                if packets:
                    pkt = packets[-1]
                    with self._lock:
                        self.frame_disp = pkt.getFrame()    # uint8 disparity

                # Poll at 20 ms – fast enough to catch 8-fps frames within 20 ms
                time.sleep(0.020)

            except Exception as e:
                if self.running:
                    print(f"[Camera] Reader error: {e}")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_latest_frames(self):
        """
        Returns a dict with the most recent frames:
            {
              'timestamp': float (Unix time),
              'rgb':   np.ndarray | None  – BGR uint8, 416×320
              'depth': np.ndarray | None  – uint16 mm, aligned to RGB
              'disp':  np.ndarray | None  – uint8, raw disparity
            }
        """
        with self._lock:
            return {
                'timestamp': time.time(),
                'rgb':       self.frame_rgb,
                'depth':     self.frame_depth,
                'disp':      self.frame_disp,
            }

    def get_depth_at_point(self, x: int, y: int) -> float:
        """
        Return depth in metres at pixel (x, y) of the aligned depth map.
        Returns -1.0 if depth is unavailable or invalid.
        """
        with self._lock:
            if self.frame_depth is None:
                return -1.0
            h, w = self.frame_depth.shape
            if not (0 <= y < h and 0 <= x < w):
                return -1.0
            mm = float(self.frame_depth[y, x])
            return mm / 1000.0 if mm > 0 else -1.0

    def stop(self):
        self.running = False