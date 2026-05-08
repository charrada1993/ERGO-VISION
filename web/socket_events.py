# web/socket_events.py
# Optimized for Jetson Orin: single camera, throttled skeleton/IMU emits,
# conditional depth masking, and numpy import fix.
import time
import math
import traceback
import numpy as np                       # ← was missing (caused NameError on depth masking)
import cv2
from flask_socketio import emit
from config import JetsonConfig


def _sanitize(obj):
    """
    Recursively convert numpy scalar / array types to plain Python primitives
    so that flask-socketio's JSON encoder never raises TypeError.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class SocketEvents:
    def __init__(self, socketio, cam_managers, pose_est, pose_fusion,
                 skeleton, rula_calc, reba_calc, logger, app):
        self.socketio     = socketio
        self.cam_managers = cam_managers[:1]   # Single camera only
        self.pose_est     = pose_est
        self.pose_fusion  = pose_fusion
        self.skeleton     = skeleton
        self.rula_calc    = rula_calc
        self.reba_calc    = reba_calc
        self.logger       = logger
        self.app          = app
        self.running      = True
        self.is_recording = False
        self._frame_count = 0

        # Cached IMU snapshot – only refreshed every N frames
        self._imu_cache     = None
        self._imu_cache_age = 0

        @socketio.on('connect')
        def handle_connect():
            print("[Socket] Client connected")
            emit('config', {'mode': 1, 'usb3': False})   # USB 2.0, 1 camera

        @socketio.on('start_recording')
        def handle_start_recording():
            print("[Socket] Start recording requested")
            filename = self.logger.start_session()
            self.is_recording = True
            emit('recording_status', {'is_recording': True, 'filename': filename, 'samples': 0})

        @socketio.on('stop_recording')
        def handle_stop_recording():
            print("[Socket] Stop recording requested")
            self.is_recording = False
            self.logger.end_session()
            emit('recording_status', {'is_recording': False, 'samples': self.logger.sample_count})

    # ──────────────────────────────────────────────────────────────────
    def process_loop(self):
        """
        Background thread for the single-camera Jetson Orin pipeline:
          1. Grab the latest RGB frame from the primary CameraManager
          2. Optional depth-based person masking (every Nth frame only)
          3. Run MediaPipe pose estimation
          4. Compute skeleton angles
          5. Calculate RULA + REBA scores
          6. Emit pose_update (every frame) and skeleton_3d (every Nth frame)
          7. Log at interval
        """
        print("[Processing] Thread started (single-camera, Jetson optimized)")
        last_log   = 0
        self._frame_count = 0

        # Precompute throttle intervals from config
        _skel_every  = JetsonConfig.SKELETON_EMIT_EVERY   # 3
        _imu_every   = JetsonConfig.IMU_SAMPLE_EVERY      # 3
        _mask_every  = JetsonConfig.DEPTH_MASK_EVERY      # 2
        _interval    = 1.0 / JetsonConfig.CAMERA_FPS      # ~0.125 s at 8 fps

        while self.running:
            try:
                # ── 1. Grab frame from the single camera ───────────────
                mgr    = self.cam_managers[0]
                frames = mgr.get_latest_frames()
                rgb    = frames.get('rgb') if frames else None

                if rgb is None:
                    if self._frame_count % 60 == 0:
                        print("[Processing] Waiting for first camera frame …")
                    time.sleep(0.05)
                    continue

                depth_frame = frames.get('depth')

                # ── 2. Depth masking (every _mask_every frames only) ────
                masked_rgb = rgb
                if depth_frame is not None and (self._frame_count % _mask_every == 0):
                    if depth_frame.shape == rgb.shape[:2]:
                        mask = ((depth_frame > 500) & (depth_frame < 3000)).astype(np.uint8)
                        masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)

                # ── 3. Pose estimation ─────────────────────────────────
                lm = self.pose_est.get_landmarks(masked_rgb)
                if lm is None:
                    if self._frame_count % 60 == 0:
                        print("[Processing] No landmarks detected")
                    time.sleep(0.05)
                    continue

                self._frame_count += 1

                # ── 4. Fuse + compute angles ──────────────────────────
                skeleton_3d = self.pose_fusion.fuse([lm])
                if skeleton_3d is None:
                    continue

                angles = self.skeleton.compute_angles(skeleton_3d)

                # Depth-enriched angles (optional)
                if depth_frame is not None and hasattr(self.skeleton, 'enrich_with_depth'):
                    angles = self.skeleton.enrich_with_depth(angles, depth_frame)

                # ── 5. RULA + REBA ────────────────────────────────────
                rula_res = self.rula_calc.compute(angles)
                reba_res = self.reba_calc.compute(angles)

                rula_details = {
                    'upper_arm':   rula_res.get('upper_arm_score'),
                    'lower_arm':   rula_res.get('lower_arm_score'),
                    'wrist':       rula_res.get('wrist_score'),
                    'wrist_twist': rula_res.get('wrist_twist', 1),
                    'neck':        rula_res.get('neck_score'),
                    'trunk':       rula_res.get('trunk_score'),
                    'legs':        rula_res.get('legs_score', 1),
                    'muscle':      rula_res.get('muscle_score', 0),
                    'activity':    rula_res.get('activity_score', 0),
                    'score_a':     rula_res.get('score_A'),
                    'score_b':     rula_res.get('score_B'),
                    'score_c':     rula_res.get('score_C'),
                }

                reba_details = {
                    'trunk':        reba_res.get('trunk_score'),
                    'trunk_mod':    reba_res.get('trunk_mod', 0),
                    'neck':         reba_res.get('neck_score'),
                    'neck_mod':     reba_res.get('neck_mod', 0),
                    'legs':         reba_res.get('legs_score'),
                    'knee_mod':     reba_res.get('knee_mod', 0),
                    'upper_arm':    reba_res.get('upper_arm_score'),
                    'shoulder_mod': reba_res.get('shoulder_mod', 0),
                    'lower_arm':    reba_res.get('lower_arm_score'),
                    'wrist':        reba_res.get('wrist_score'),
                    'wrist_twist':  reba_res.get('wrist_twist', 0),
                    'coupling':     reba_res.get('coupling', 0),
                    'table_a':      reba_res.get('table_A'),
                    'table_b':      reba_res.get('table_B'),
                    'score_c':      reba_res.get('score_C'),
                }

                # ── 6. Anomaly detection ──────────────────────────────
                anomalies = []
                neck_angle  = angles.get('neck', 0)
                trunk_angle = angles.get('trunk', 0)
                if abs(neck_angle) > 40:
                    anomalies.append(f"Neck flexion: {neck_angle:.1f}° (>40°)")
                if abs(trunk_angle) > 60:
                    anomalies.append(f"Trunk forward lean: {trunk_angle:.1f}° (>60°)")
                ua_left  = angles.get('upper_arm_left', 0)
                ua_right = angles.get('upper_arm_right', 0)
                if ua_left > 90 or ua_right > 90:
                    anomalies.append("Shoulder elevated above 90°")

                # ── 7. IMU data (cached – refreshed every _imu_every frames) ──
                if self._frame_count % _imu_every == 0 or self._imu_cache is None:
                    self._imu_cache = self._get_imu_data()

                # ── 8. Emit pose_update (every frame) ────────────────
                payload = {
                    'angles':       angles,
                    'rula':         rula_res.get('RULA_score', 0),
                    'reba':         reba_res.get('REBA_score', 0),
                    'risk_level':   rula_res.get('risk_level', 'Low'),
                    'anomalies':    anomalies,
                    'rula_details': rula_details,
                    'reba_details': reba_details,
                    'imu':          self._imu_cache,
                }
                if self.is_recording:
                    payload['recording'] = {
                        'is_recording': True,
                        'samples': self.logger.sample_count
                    }

                self.socketio.emit('pose_update', _sanitize(payload))

                # ── 9. Emit skeleton_3d (every _skel_every frames) ────
                # 33 × 3 floats per emit – not needed at full frame rate.
                if self._frame_count % _skel_every == 0 and skeleton_3d is not None:
                    self.socketio.emit('skeleton_3d', {
                        'landmarks': skeleton_3d.tolist()
                        if hasattr(skeleton_3d, 'tolist') else skeleton_3d
                    })

                # ── 10. Logging ───────────────────────────────────────
                now = time.time()
                should_log = self.is_recording or (now - last_log >= 0.5)
                if should_log:
                    try:
                        from ergonomics.risk import RiskAnalyzer
                        vision_anoms = RiskAnalyzer.detect_anomalies(
                            angles,
                            rula_res.get('RULA_score', 0),
                            reba_res.get('REBA_score', 0)
                        )
                        self.logger.log(angles, rula_res, reba_res,
                                        vision_anoms + anomalies)
                    except Exception as log_err:
                        print(f"[Processing] Logger error: {log_err}")
                    if not self.is_recording:
                        last_log = now

            except Exception as e:
                print(f"[Processing] ERROR: {e}")
                traceback.print_exc()
                # Prevent tight busy-loop on repeated errors
                time.sleep(_interval)
                continue

            # Pace the loop to match camera frame rate
            time.sleep(_interval)

    # ------------------------------------------------------------------
    def _get_imu_data(self):
        """Safely read from the Visual IMU manager."""
        try:
            imu_mgr = self.app.config.get('IMU_MANAGER')
            if imu_mgr is None:
                return None
            d = imu_mgr.get_data()
            roll, pitch, yaw = d.get('euler', (0.0, 0.0, 0.0))
            ax, ay, az       = d.get('accel', (0.0, 0.0, 0.0))
            gx, gy, gz       = d.get('gyro',  (0.0, 0.0, 0.0))
            return {
                'roll':  round(roll,  2),
                'pitch': round(pitch, 2),
                'yaw':   round(yaw,   2),
                'accel': {'x': round(ax, 4), 'y': round(ay, 4), 'z': round(az, 4)},
                'gyro':  {'x': round(gx, 4), 'y': round(gy, 4), 'z': round(gz, 4)},
            }
        except Exception:
            return None