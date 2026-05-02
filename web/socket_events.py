# web/socket_events.py
import time
import math
import traceback
from flask_socketio import emit


class SocketEvents:
    def __init__(self, socketio, cam_managers, pose_est, pose_fusion,
                 skeleton, rula_calc, reba_calc, logger, app):
        self.socketio   = socketio
        self.cam_managers = cam_managers
        self.pose_est   = pose_est
        self.pose_fusion = pose_fusion
        self.skeleton   = skeleton
        self.rula_calc  = rula_calc
        self.reba_calc  = reba_calc
        self.logger     = logger
        self.app        = app
        self.running    = True
        self.is_recording = False
        self._frame_count = 0

        @socketio.on('connect')
        def handle_connect():
            print("[Socket] Client connected")
            mode = len(self.cam_managers)
            emit('config', {'mode': mode, 'usb3': True})

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
        Background thread:
          1. Grab the latest RGB frames from all CameraManagers
          2. Run MediaPipe / pose estimator on each
          3. Fuse landmarks and compute skeleton angles
          4. Calculate RULA + REBA with sub-score details
          5. Evaluate vision-based anomalies
          6. Emit 'pose_update' and 'skeleton_3d' to clients
          7. Log every 0.5 s
        """
        print("[Processing] Thread started")
        last_log   = 0
        self._frame_count = 0

        while self.running:
            try:
                # ── 1. Get latest frames & estimate poses ──────────────────
                all_landmarks = []
                depth_frame = None

                for mgr in self.cam_managers:
                    frames = mgr.get_latest_frames()
                    rgb    = frames.get('rgb') if frames else None
                    if frames and frames.get('depth') is not None and depth_frame is None:
                        # Grab depth frame from the first available camera
                        depth_frame = frames.get('depth')

                    if rgb is not None:
                        # OPTION 1: Depth-Based Masking
                        # Only process pixels within the "working zone" (e.g., 0.5m to 3.0m)
                        # This prevents MediaPipe from latching onto multiple people in the background
                        masked_rgb = rgb
                        if frames.get('depth') is not None:
                            depth = frames.get('depth')
                            if depth.shape == rgb.shape[:2]:
                                mask = (depth > 500) & (depth < 3000)
                                masked_rgb = rgb.copy()
                                masked_rgb[~mask] = 0
                                
                        lm = self.pose_est.get_landmarks(masked_rgb)
                        all_landmarks.append(lm)
                    else:
                        all_landmarks.append(None)
                
                # Check if at least one camera provided landmarks
                if not any(lm is not None for lm in all_landmarks):
                    if self._frame_count % 60 == 0:
                        print("[Processing] No landmarks detected from any camera")
                    time.sleep(0.05)
                    continue

                self._frame_count += 1

                # ── 2. Fuse + compute angles ──────────────────────────
                skeleton_3d = self.pose_fusion.fuse(all_landmarks)
                if skeleton_3d is None:
                    continue

                angles      = self.skeleton.compute_angles(skeleton_3d)

                # Enrich depth-aware angles if depth available
                if depth_frame is not None and hasattr(self.skeleton, 'enrich_with_depth'):
                    angles = self.skeleton.enrich_with_depth(angles, depth_frame)

                # ── 3. RULA + REBA scores with details ────────────────
                rula_res = self.rula_calc.compute(angles)
                reba_res = self.reba_calc.compute(angles)

                # Build sub-score detail dicts for the UI
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

                # ── 4. Vision-based angle anomalies ─────────────────
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

                # ── 5. Emit Socket.IO update ──────────────────────────
                payload = {
                    'angles':       angles,
                    'rula':         rula_res.get('RULA_score', 0),
                    'reba':         reba_res.get('REBA_score', 0),
                    'risk_level':   rula_res.get('risk_level', 'Low'),
                    'anomalies':    anomalies,
                    'rula_details': rula_details,
                    'reba_details': reba_details,
                    'imu':          self._get_imu_data(),
                }
                if self.is_recording:
                    payload['recording'] = {
                        'is_recording': True,
                        'samples': self.logger.sample_count
                    }

                self.socketio.emit('pose_update', payload)

                if skeleton_3d is not None:
                    self.socketio.emit('skeleton_3d', {
                        'landmarks': skeleton_3d.tolist() if hasattr(skeleton_3d, 'tolist') else skeleton_3d
                    })

                # ── 6. Logging (Continuous if recording, periodic otherwise) ──
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

            # ~10 Hz processing rate
            time.sleep(1.0 / 10)

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