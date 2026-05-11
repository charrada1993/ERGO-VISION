# pose/skeleton.py
import numpy as np

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

class SkeletonBuilder:
    def __init__(self):
        # Aspect ratio for coordinate correction (1280x720 = 16:9)
        self.aspect_ratio = 16.0 / 9.0
        # EMA smoothing factor (0.0 to 1.0; lower = smoother, more lag)
        self.alpha = 0.4
        self._last_angles = {}

    @staticmethod
    def compute_euler(v):
        """
        Compute Pitch (flexion/extension) and Roll (abduction/adduction).
        MediaPipe coordinate system: X right, Y down, Z forward.
        """
        vx, vy, vz = v
        # Pitch (Flexion/Extension): sagittal plane (Y-Z)
        pitch = np.degrees(np.arctan2(vz, vy))
        # Roll (Abduction/Adduction): coronal plane (X-Y)
        roll = np.degrees(np.arctan2(vx, vy))
        return pitch, roll

    def compute_angles(self, landmarks_3d):
        """
        Compute joint angles with Aspect Ratio Correction.
        landmarks_3d: 33x3 normalized [x, y, z] from MediaPipe.
        """
        if landmarks_3d is None or len(landmarks_3d) < 33:
            return {}

        # ── 1. Coordinate Correction ─────────────────────────────────────
        # Normalize landmarks into a square space to fix 16:9 distortion.
        # Per MediaPipe docs, 'z' uses roughly the same scale as 'x'.
        lm = landmarks_3d.copy()
        lm[:, 0] *= self.aspect_ratio
        lm[:, 2] *= self.aspect_ratio

        angles = {}
        
        # ── 2. Axial segments ────────────────────────────────────────────
        # Neck angle (Shoulder midpoint to Nose)
        shl_mid = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
        neck_vec = lm[NOSE] - shl_mid
        neck_pitch, neck_roll = self.compute_euler(-neck_vec)
        angles['neck'] = neck_pitch
        angles['neck_mod'] = 1 if abs(neck_roll) > 15 else 0

        # Trunk angle (Hip midpoint to Shoulder midpoint)
        hip_mid = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
        trunk_vec = shl_mid - hip_mid
        trunk_pitch, trunk_roll = self.compute_euler(-trunk_vec)
        angles['trunk'] = trunk_pitch
        angles['trunk_mod'] = 1 if abs(trunk_roll) > 15 else 0

        # ── 3. Appendages (Left side primary for RULA/REBA) ──────────────
        # Upper arm (shoulder to elbow)
        ua_vec = lm[LEFT_ELBOW] - lm[LEFT_SHOULDER]
        arm_pitch, arm_roll = self.compute_euler(ua_vec)
        angles['upper_arm_left'] = arm_pitch
        angles['shoulder_mod'] = 1 if abs(arm_roll) > 20 else 0

        # Forearm (elbow to wrist)
        fa_vec = lm[LEFT_WRIST] - lm[LEFT_ELBOW]
        cos = np.dot(ua_vec, fa_vec) / (np.linalg.norm(ua_vec) * np.linalg.norm(fa_vec) + 1e-6)
        angles['elbow_left'] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        # Wrist Heuristic (MediaPipe Pose lacks hand joints, but has pinky/index)
        # Flexion approximated by hand vector (wrist to mid-hand) relative to forearm
        hand_mid = (lm[17] + lm[19]) / 2.0 # Left pinky + Left index
        hand_vec = hand_mid - lm[LEFT_WRIST]
        cos_w = np.dot(fa_vec, hand_vec) / (np.linalg.norm(fa_vec) * np.linalg.norm(hand_vec) + 1e-6)
        angles['wrist_left'] = np.degrees(np.arccos(np.clip(cos_w, -1.0, 1.0)))

        # Legs stability (assume stable if standing)
        angles['legs_stable'] = True

        # ── 4. Right side ───────────────────────────────────────────────
        ua_vec_r = lm[RIGHT_ELBOW] - lm[RIGHT_SHOULDER]
        arm_pitch_r, _ = self.compute_euler(ua_vec_r)
        angles['upper_arm_right'] = arm_pitch_r
        
        fa_vec_r = lm[RIGHT_WRIST] - lm[RIGHT_ELBOW]
        cos_r = np.dot(ua_vec_r, fa_vec_r) / (np.linalg.norm(ua_vec_r) * np.linalg.norm(fa_vec_r) + 1e-6)
        angles['elbow_right'] = np.degrees(np.arccos(np.clip(cos_r, -1.0, 1.0)))

        return angles

    def enrich_with_depth(self, angles, depth_frame, calib=None):
        """
        Provides temporal smoothing (EMA) and eventually metric back-projection.
        Currently ensures stable angles for RULA/REBA scoring.
        """
        # Temporal smoothing (EMA) to stabilize scores
        if not self._last_angles:
            self._last_angles = angles
            return angles
            
        smoothed = {}
        for k, v in angles.items():
            if isinstance(v, (int, float)) and k in self._last_angles:
                # Apply 40% current / 60% history EMA
                smoothed[k] = self.alpha * v + (1 - self.alpha) * self._last_angles[k]
            else:
                smoothed[k] = v
        
        self._last_angles = smoothed
        return smoothed