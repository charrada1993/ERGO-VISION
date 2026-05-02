# camera/imu_manager.py
# Visual IMU – derives orientation & motion from RGB camera optical flow.
# No hardware IMU chip required.  Works with any OAK-D variant (including Lite).
#
# Approach:
#   1. Lucas-Kanade sparse optical flow tracks good feature points frame-to-frame.
#   2. The average flow vector gives translational (pseudo-acceleration) motion.
#   3. A homography estimated from the flow gives the rotational component
#      which is decomposed into roll / pitch / yaw estimates.
#   4. All values are smoothed with a lightweight exponential moving average.
#
# Output dict matches the shape previously expected by the rest of the system
# so no changes to socket_events.py or downstream consumers are needed.

import cv2
import numpy as np
import threading
import time
import math


# ── Tuning constants ─────────────────────────────────────────────────────────
_LK_PARAMS = dict(
    winSize   = (21, 21),
    maxLevel  = 3,
    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_FEATURE_PARAMS = dict(
    maxCorners   = 200,
    qualityLevel = 0.01,
    minDistance  = 10,
    blockSize    = 7,
)
_REDETECT_INTERVAL = 10   # re-detect features every N frames
_EMA_ALPHA         = 0.3  # smoothing factor (0 = no update, 1 = no smoothing)
_SCALE_ACCEL       = 0.02 # pixels/frame → pseudo m/s²  (display scale, not physical)
_SCALE_GYRO        = 0.5  # rotation deg/frame → pseudo rad/s


class IMUManager:
    """
    Visual IMU: uses the RGB camera frames to produce IMU-like telemetry
    (roll, pitch, yaw, pseudo-accelerometer, pseudo-gyroscope) without
    any physical IMU sensor.

    Drop-in replacement for the hardware IMUManager.  The public API
    (setup / start / get_data / stop) is identical.
    """

    def __init__(self, pipeline=None, device=None):
        # pipeline / device kept for API compatibility; not used here
        self.pipeline = pipeline
        self.device   = device
        self.running  = False
        self._lock    = threading.Lock()

        # Caller provides the CameraManager so we can pull RGB frames
        self._cam_mgr = None

        # Internal optical-flow state
        self._prev_gray  = None
        self._prev_pts   = None
        self._frame_idx  = 0

        # Smoothed outputs
        self._roll  = 0.0
        self._pitch = 0.0
        self._yaw   = 0.0
        self._ax    = 0.0   # pseudo-accel x (m/s² equivalent)
        self._ay    = 0.0
        self._gx    = 0.0   # pseudo-gyro x (rad/s equivalent)
        self._gy    = 0.0
        self._gz    = 0.0

        self.latest = self._make_snapshot()
        self.callback = None

    # ------------------------------------------------------------------
    # Public setup API (matches hardware IMUManager)
    # ------------------------------------------------------------------
    def setup(self):
        """No pipeline nodes needed for visual IMU."""
        print("[VisualIMU] Visual IMU ready – no hardware sensor required.")
        return True

    def set_camera_manager(self, cam_mgr):
        """Provide the CameraManager whose RGB frames we will process."""
        self._cam_mgr = cam_mgr

    def start(self, callback=None):
        """Launch the background optical-flow thread."""
        if self._cam_mgr is None:
            print("[VisualIMU] ERROR – call set_camera_manager() first.")
            return
        self.callback = callback
        self.running  = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        print("[VisualIMU] Streaming started (optical-flow mode)")

    # ------------------------------------------------------------------
    # Background optical-flow thread
    # ------------------------------------------------------------------
    def _reader(self):
        while self.running:
            try:
                frames = self._cam_mgr.get_latest_frames()
                rgb    = frames.get('rgb') if frames else None
                if rgb is None:
                    time.sleep(0.02)
                    continue

                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                self._process_frame(gray)

                snapshot = self._make_snapshot()
                with self._lock:
                    self.latest = snapshot

                if self.callback:
                    self.callback(self.get_data())

            except Exception as e:
                if self.running:
                    print(f"[VisualIMU] Error: {e}")

            time.sleep(1.0 / 15)   # Match the 15 FPS camera rate

    # ------------------------------------------------------------------
    # Core optical-flow processing
    # ------------------------------------------------------------------
    def _process_frame(self, gray: np.ndarray):
        self._frame_idx += 1

        # ── First frame or re-detect features ───────────────────────────
        if (self._prev_gray is None or
                self._prev_pts is None or
                len(self._prev_pts) < 20 or
                self._frame_idx % _REDETECT_INTERVAL == 0):
            pts = cv2.goodFeaturesToTrack(gray, mask=None, **_FEATURE_PARAMS)
            if pts is None:
                self._prev_gray = gray
                return
            self._prev_pts  = pts
            self._prev_gray = gray
            return

        # ── Lucas-Kanade optical flow ────────────────────────────────────
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **_LK_PARAMS
        )

        good_prev = self._prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) < 8:
            # Too few points – reset
            self._prev_gray = gray
            self._prev_pts  = None
            return

        # ── Translational flow → pseudo-accelerometer ────────────────────
        flow = good_next - good_prev           # (N, 2) pixel displacement
        mean_flow = flow.mean(axis=0)          # [dx, dy]

        ax_raw = float(mean_flow[0]) * _SCALE_ACCEL
        ay_raw = float(mean_flow[1]) * _SCALE_ACCEL

        # ── Homography → rotational component ────────────────────────────
        H, mask = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 3.0)

        roll_delta = pitch_delta = yaw_delta = 0.0
        gz_raw     = 0.0

        if H is not None:
            # In-plane rotation from homography → pseudo yaw
            angle_rad  = math.atan2(H[1, 0], H[0, 0])
            gz_raw     = angle_rad * _SCALE_GYRO
            yaw_delta  = math.degrees(angle_rad)
        else:
            gz_raw    = 0.0
            yaw_delta = 0.0

        # ── Exact formulas from reference screenshot ──────────────────────
        # Pitch (θ) = atan2(ax, sqrt(ay² + az²)) × 180 / π
        # Roll  (φ) = atan2(ay, sqrt(ax² + az²)) × 180 / π
        # ax/ay derived from mean optical-flow displacement; az≈0 (2-D camera)
        ax_raw = float(mean_flow[0]) * _SCALE_ACCEL
        ay_raw = float(mean_flow[1]) * _SCALE_ACCEL
        az_raw = 0.0

        pitch_raw = math.degrees(
            math.atan2(ax_raw, math.sqrt(ay_raw ** 2 + az_raw ** 2))
        )
        roll_raw  = math.degrees(
            math.atan2(ay_raw, math.sqrt(ax_raw ** 2 + az_raw ** 2))
        )

        # ── Exponential moving average (smoothing) ───────────────────────
        a = _EMA_ALPHA
        self._ax    = a * ax_raw    + (1 - a) * self._ax
        self._ay    = a * ay_raw    + (1 - a) * self._ay
        self._gz    = a * gz_raw    + (1 - a) * self._gz
        self._roll  = a * roll_raw  + (1 - a) * self._roll
        self._pitch = a * pitch_raw + (1 - a) * self._pitch
        self._yaw  += a * yaw_delta   # integrate yaw

        # ── Update tracking state ────────────────────────────────────────
        self._prev_gray = gray
        self._prev_pts  = good_next.reshape(-1, 1, 2)

    # ------------------------------------------------------------------
    # Snapshot helper
    # ------------------------------------------------------------------
    def _make_snapshot(self) -> dict:
        # Build a unit quaternion from roll/pitch/yaw for compatibility
        quat = self._euler_to_quat(self._roll, self._pitch, self._yaw)
        return {
            'accel':           (self._ax, self._ay, 0.0),
            'gyro':            (self._gx, self._gy, self._gz),
            'rotation_vector': quat,          # (i, j, k, real)
            'accuracy':        0.0,
            'accel_ts_ms':     0.0,
            'gyro_ts_ms':      0.0,
            'rv_ts_ms':        0.0,
            'timestamp':       time.time(),
            'euler':           (self._roll, self._pitch, self._yaw),
        }

    # ------------------------------------------------------------------
    # Public accessors (matches hardware IMUManager API)
    # ------------------------------------------------------------------
    def get_data(self) -> dict:
        with self._lock:
            return dict(self.latest)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
        """Roll/pitch/yaw (degrees) → unit quaternion (i, j, k, real)."""
        r = math.radians(roll_deg)  / 2.0
        p = math.radians(pitch_deg) / 2.0
        y = math.radians(yaw_deg)   / 2.0

        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)

        real = cr * cp * cy + sr * sp * sy
        i    = sr * cp * cy - cr * sp * sy
        j    = cr * sp * cy + sr * cp * sy
        k    = cr * cp * sy - sr * sp * cy
        return (i, j, k, real)

    def stop(self):
        self.running = False