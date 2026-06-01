# web/socket_events.py
# Optimized for Jetson Orin: single camera, throttled skeleton/IMU emits,
# conditional depth masking, GIL-yielding sleep, and numpy import fix.
import time
import math
import traceback
import threading
import numpy as np                       # ← was missing (caused NameError on depth masking)
import cv2
from flask_socketio import emit
from config import JetsonConfig

# Pre-import RiskAnalyzer at module load time (avoids repeated import overhead in loop)
try:
    from ergonomics.risk import RiskAnalyzer as _RiskAnalyzer
except ImportError:
    _RiskAnalyzer = None

try:
    from ai.operation.inference import NumpyInference
except ImportError:
    NumpyInference = None


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

        # AI Model initialization
        self.ai_model = None
        if NumpyInference:
            try:
                self.ai_model = NumpyInference()
            except Exception as e:
                print(f"[Socket] AI Model load failed: {e}")

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
            import os
            filename = os.path.basename(self.logger.session_path) if getattr(self.logger, 'session_path', None) else ""
            self.logger.end_session()
            emit('recording_status', {'is_recording': False, 'samples': self.logger.sample_count, 'filename': filename})

    def process_loop(self):
        """
        Background thread for the single-camera Jetson Orin pipeline:
          1. Grab the latest RGB frame from the primary CameraManager
          2. Optional depth-based person masking (every Nth frame only)
          3. Run MediaPipe pose estimation
          4. Compute skeleton angles
          5. Calculate RULA + REBA scores
          6. Emit pose_update (throttled, max 5 Hz) and skeleton_3d (every Nth frame)
          7. Log at interval (1 Hz, or every sample if recording)
        """
        print("[Processing] Thread started (single-camera, Jetson optimized)")
        last_log        = 0.0
        last_emit       = 0.0
        last_status_msg = 0.0        # time-based gate for status prints
        self._frame_count = 0

        # Precompute throttle intervals from config
        _skel_every  = JetsonConfig.SKELETON_EMIT_EVERY   # 4
        _imu_every   = JetsonConfig.IMU_SAMPLE_EVERY      # 5
        _mask_every  = JetsonConfig.DEPTH_MASK_EVERY      # 3
        _min_emit_dt = 1.0 / JetsonConfig.POSE_UPDATE_MAX_HZ   # 0.2 s at 5 Hz
        _loop_sleep  = JetsonConfig.PROCESS_LOOP_SLEEP    # 5 ms GIL yield
        _log_interval = 1.0                               # log at 1 Hz (not 2 Hz)
        _STATUS_DT   = 5.0                                # print status at most every 5 s
        _last_rgb    = None                               # skip re-processing same frame
        _interval    = 0.125                              # fallback sleep (8 fps period)

        # ── Score hysteresis: only change displayed score after N stable frames ──
        # Prevents RULA/REBA flipping at boundary values (e.g. neck exactly 20°).
        _SCORE_HOLD  = 3            # frames a new score must persist before accepted
        _rula_stable = 0            # frames current candidate has held
        _reba_stable = 0
        _rula_cand   = None         # candidate score (not yet committed)
        _reba_cand   = None
        _rula_committed = None      # last committed (displayed) score
        _reba_committed = None
        _rula_details_committed = {}
        _reba_details_committed = {}
        _risk_committed = 'Low'
        # Last good payload angles/anomalies (dropout holdout at emit layer)
        _last_angles   = {}
        _last_anomalies = []

        while self.running:
            try:
                now = time.time()

                # ── 1. Grab frame from the single camera ───────────────
                mgr    = self.cam_managers[0]
                frames = mgr.get_latest_frames()
                rgb    = frames.get('rgb') if frames else None

                if rgb is None:
                    if now - last_status_msg >= _STATUS_DT:
                        print("[Processing] Waiting for first camera frame …")
                        last_status_msg = now
                    time.sleep(0.020)
                    continue

                # Skip if same frame object as last iteration (no new camera data)
                if rgb is _last_rgb:
                    time.sleep(0.010)
                    continue
                _last_rgb = rgb

                depth_frame = frames.get('depth')

                # ── 2. Resize + Depth masking (stable landmarks, every frame) ──
                # Optimization: Resize to MediaPipe resolution (320x180) BEFORE masking
                # This saves massive CPU compared to masking at 720p.
                _inf_w = JetsonConfig.POSE_INPUT_WIDTH   # 320
                _inf_h = JetsonConfig.POSE_INPUT_HEIGHT  # 180
                
                small_rgb = cv2.resize(rgb, (_inf_w, _inf_h), interpolation=cv2.INTER_LINEAR)
                
                if depth_frame is not None:
                    # Resize depth to match small_rgb
                    small_depth = cv2.resize(depth_frame, (_inf_w, _inf_h), interpolation=cv2.INTER_NEAREST)
                    # Create mask (person within 0.5m to 3.5m)
                    mask = ((small_depth > 500) & (small_depth < 3500)).astype(np.uint8)
                    masked_small_rgb = cv2.bitwise_and(small_rgb, small_rgb, mask=mask)
                else:
                    masked_small_rgb = small_rgb

                # ── 3. Pose estimation & dynamic mock fallback ────────
                from camera.mock_manager import MockCameraManager
                is_mock = isinstance(mgr, MockCameraManager)

                if is_mock:
                    # Generate dynamic, realistic synthetic joint angles
                    t = time.time()
                    angles = {
                        'neck':            15.0 + math.sin(t) * 10.0,
                        'trunk':           20.0 + math.cos(t * 0.8) * 15.0,
                        'upper_arm_left':  45.0 + math.sin(t * 1.2) * 25.0,
                        'upper_arm_right': 30.0 + math.cos(t * 1.5) * 20.0,
                        'elbow_left':      90.0 + math.sin(t * 0.9) * 30.0,
                        'elbow_right':     80.0 + math.cos(t * 1.1) * 25.0,
                        'wrist_left':      10.0 + math.sin(t * 1.4) * 15.0,
                        'wrist_right':     12.0 + math.cos(t * 1.3) * 12.0,
                        'hip_left':        15.0 + math.sin(t * 0.5) * 5.0,
                        'hip_right':       15.0 + math.cos(t * 0.5) * 5.0,
                        'knee_left':       5.0 + math.sin(t * 0.7) * 5.0,
                        'knee_right':      5.0 + math.cos(t * 0.7) * 5.0,
                    }
                    lm = np.zeros((33, 3), dtype=np.float32)
                    for i in range(33):
                        lm[i] = [0.5 + 0.1 * math.sin(t + i), 0.5 + 0.1 * math.cos(t + i), 0.0]
                    
                    skeleton_3d = np.zeros((33, 3), dtype=np.float32)
                    for i in range(33):
                        skeleton_3d[i] = [0.0, 0.0, 1.5]
                    
                    self._frame_count += 1
                else:
                    lm = self.pose_est.get_landmarks(masked_small_rgb)
                    if lm is None:
                        # ── Dropout path: hold last good scores for up to _MAX_DROPOUT ──
                        hold = self.skeleton.enrich_with_depth({})   # returns holdout or {}
                        if hold and _rula_committed is not None:
                            # Re-emit with last committed scores so UI doesn't flash '--'
                            now2 = time.time()
                            if now2 - last_emit >= _min_emit_dt:
                                self.socketio.emit('pose_update', _sanitize({
                                    'angles':       _last_angles,
                                    'rula':         _rula_committed,
                                    'reba':         _reba_committed,
                                    'risk_level':   _risk_committed,
                                    'anomalies':    _last_anomalies,
                                    'rula_details': _rula_details_committed,
                                    'reba_details': _reba_details_committed,
                                    'imu':          self._imu_cache,
                                    'ai_results':   None,
                                    '_holdout':     True,
                                }))
                                last_emit = now2
                        else:
                            if now - last_status_msg >= _STATUS_DT:
                                print("[Processing] No person detected in frame")
                                last_status_msg = now
                        time.sleep(0.020)
                        continue

                    self._frame_count += 1

                    # ── 4. Fuse + compute angles ──────────────────────────
                    skeleton_3d = self.pose_fusion.fuse([lm])
                    if skeleton_3d is None:
                        continue

                    # ── 5. RULA + REBA — pass depth + calib so 3D pipeline is active ──
                    calib = self.app.config.get('CALIBRATION')
                    frame_h, frame_w = rgb.shape[:2]
                    angles = self.skeleton.compute_angles(
                        skeleton_3d,
                        calib=calib,
                        depth_frame=depth_frame,   # uint16 mm, aligned to RGB
                        world_landmarks=None,       # fallback handled inside compute_angles
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                    # EMA smoothing pass
                    angles = self.skeleton.enrich_with_depth(angles)

                rula_res = self.rula_calc.compute(angles)
                reba_res = self.reba_calc.compute(angles)

                # ── Score hysteresis ───────────────────────────────────
                # A score only becomes the "committed" (displayed) value after
                # staying the same for _SCORE_HOLD consecutive frames.
                new_rula = rula_res.get('RULA_score', 0)
                new_reba = reba_res.get('REBA_score', 0)

                if new_rula == _rula_cand:
                    _rula_stable += 1
                else:
                    _rula_cand, _rula_stable = new_rula, 1

                if new_reba == _reba_cand:
                    _reba_stable += 1
                else:
                    _reba_cand, _reba_stable = new_reba, 1

                if _rula_stable >= _SCORE_HOLD or _rula_committed is None:
                    _rula_committed          = _rula_cand
                    _rula_details_committed  = {
                        'upper_arm':   rula_res.get('upper_arm_score'),
                        'lower_arm':   rula_res.get('lower_arm_score'),
                        'wrist':       rula_res.get('wrist_score'),
                        'wrist_twist': rula_res.get('wrist_twist', 1),
                        'neck':        rula_res.get('neck_score'),
                        'trunk':       rula_res.get('trunk_score'),
                        'legs':        rula_res.get('legs_score', 1),
                        'score_a':     rula_res.get('score_A'),
                        'score_b':     rula_res.get('score_B'),
                        'score_c':     rula_res.get('score_C'),
                    }
                    _risk_committed = rula_res.get('risk_level', 'Low')

                if _reba_stable >= _SCORE_HOLD or _reba_committed is None:
                    _reba_committed          = _reba_cand
                    _reba_details_committed  = {
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
                        'table_a':      reba_res.get('table_A'),
                        'table_b':      reba_res.get('table_B'),
                        'score_c':      reba_res.get('score_C'),
                    }

                # ── 5b. AI Inference (Version 2) ──────────────────────
                ai_results = None
                if self.ai_model:
                    try:
                        if hasattr(self.ai_model, 'version') and self.ai_model.version == '2.0':
                            ai_results = self.ai_model.predict(angles)
                        else:
                            ai_results = self.ai_model.predict(lm)
                    except Exception as e:
                        print(f"[AI] Inference error: {e}")

                # ── 6. Anomaly detection ──────────────────────────────
                anomalies = []
                neck_angle  = angles.get('neck', 0)
                trunk_angle = angles.get('trunk', 0)
                knee_l = angles.get('knee_left',  0)
                knee_r = angles.get('knee_right', 0)
                wrist_l = angles.get('wrist_left', 0)
                wrist_r = angles.get('wrist_right', 0)
                if neck_angle > 40:
                    anomalies.append(f"Neck forward flexion: {neck_angle:.1f}° (>40°)")
                elif neck_angle < -15:
                    anomalies.append(f"Neck extension: {neck_angle:.1f}° (<-15°)")
                if trunk_angle > 60:
                    anomalies.append(f"Trunk forward lean: {trunk_angle:.1f}° (>60°)")
                elif trunk_angle < -10:
                    anomalies.append(f"Trunk extension: {trunk_angle:.1f}°")
                ua_left  = angles.get('upper_arm_left', 0)
                ua_right = angles.get('upper_arm_right', 0)
                if ua_left > 90 or ua_right > 90:
                    anomalies.append(f"Shoulder elevated above 90° (L:{ua_left:.0f}° R:{ua_right:.0f}°)")
                if max(knee_l, knee_r) > 60:
                    anomalies.append(f"Deep knee flexion: {max(knee_l, knee_r):.0f}° (>60°)")
                if abs(wrist_l) > 20 or abs(wrist_r) > 20:
                    anomalies.append(f"Wrist deviation: L {wrist_l:.0f}° R {wrist_r:.0f}°")

                # ── 7. IMU data (cached – refreshed every _imu_every frames) ──
                if self._frame_count % _imu_every == 0 or self._imu_cache is None:
                    self._imu_cache = self._get_imu_data()

                # ── 8. Emit pose_update (throttled to 5 Hz max) ───────
                now = time.time()
                if now - last_emit >= _min_emit_dt:
                    _last_angles   = angles
                    _last_anomalies = anomalies
                    payload = {
                        'angles':       angles,
                        'rula':         _rula_committed,
                        'reba':         _reba_committed,
                        'risk_level':   _risk_committed,
                        'anomalies':    anomalies,
                        'rula_details': _rula_details_committed,
                        'reba_details': _reba_details_committed,
                        'imu':          self._imu_cache,
                        'ai_results':   ai_results,
                    }
                    if self.is_recording:
                        payload['recording'] = {
                            'is_recording': True,
                            'samples': self.logger.sample_count
                        }
                    self.socketio.emit('pose_update', _sanitize(payload))
                    last_emit = now
                    # Yield GIL after emit so Flask/SocketIO threads can run
                    time.sleep(_loop_sleep)

                # ── 9. Emit skeleton_3d (every _skel_every frames) ────
                if self._frame_count % _skel_every == 0 and skeleton_3d is not None:
                    self.socketio.emit('skeleton_3d', {
                        'landmarks': skeleton_3d.tolist()
                        if hasattr(skeleton_3d, 'tolist') else skeleton_3d
                    })

                # ── 10. Logging (1 Hz background, or every frame if recording) ──
                now = time.time()
                should_log = self.is_recording or (now - last_log >= _log_interval)
                if should_log:
                    try:
                        if _RiskAnalyzer is not None:
                            vision_anoms = _RiskAnalyzer.detect_anomalies(
                                angles,
                                rula_res.get('RULA_score', 0),
                                reba_res.get('REBA_score', 0)
                            )
                        else:
                            vision_anoms = []
                        self.logger.log(angles, rula_res, reba_res,
                                        vision_anoms + anomalies, ai_results=ai_results)
                    except Exception:
                        pass   # logging errors are non-critical
                    if not self.is_recording:
                        last_log = now

                # ── No extra sleep here — MediaPipe inference time
                # (~50–120 ms on ARM Lite model) is the natural frame limiter.

            except Exception as e:
                print(f"[Processing] ERROR: {e}")
                # Prevent tight busy-loop on repeated errors
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