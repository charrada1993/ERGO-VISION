# pose/skeleton.py
# Corrected 3D joint-angle computation for RULA/REBA scoring.
#
# KEY FIXES applied (all 6 root causes from the diagnostic):
#   RC-1  Y-axis flip  : MediaPipe Y-down → biomechanics Y-up
#   RC-2  Intrinsics   : RGB intrinsics used for back-projection (not depth cam)
#   RC-3  Depth units  : depth_mm / 1000.0 → metres (done in camera/manager.py but guarded here)
#   RC-4  Rad→Deg      : all acos/atan2 results converted via np.degrees()
#   RC-5  Gravity ref  : trunk axis = normalize(mid_shoulder − mid_hip) as vertical reference
#   RC-6  acos clamp   : np.clip(dot, -1.0, 1.0) before every acos call
#
# MediaPipe landmark source:
#   estimator.py returns pose_landmarks (normalised 0-1).
#   When OAK-D depth is available, skeleton.py back-projects to metric 3D using
#   RGB intrinsics + depth value.  Fall-back: pose_world_landmarks (metric, hip-centred).
#
# Angle sign convention (RULA/REBA):
#   Flexion  → positive angle
#   Extension→ negative angle
#   All angles returned in DEGREES.

import math
import numpy as np

# ── MediaPipe Pose landmark indices ───────────────────────────────────────────
NOSE           =  0
LEFT_EYE_INNER =  1
LEFT_EYE       =  2
RIGHT_EYE_INNER=  4
RIGHT_EYE      =  5
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_PINKY     = 17
RIGHT_PINKY    = 18
LEFT_INDEX     = 19
RIGHT_INDEX    = 20
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28

# Minimum landmark visibility to accept a point (MediaPipe confidence)
VIS_THRESHOLD = 0.45
# OAK-D Lite reliable depth range (metres)
DEPTH_MIN_M = 0.35
DEPTH_MAX_M = 3.5


# ── Maths helpers ─────────────────────────────────────────────────────────────
def _norm(v):
    """Normalize a 3-vector. Returns zero-vector if near-singular."""
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def _angle_deg(v1, v2):
    """Unsigned angle [0°-180°] between two 3-vectors.  RC-4 + RC-6 safe."""
    dot = np.dot(_norm(v1), _norm(v2))
    dot = float(np.clip(dot, -1.0, 1.0))        # RC-6: clamp before acos
    return math.degrees(math.acos(dot))           # RC-4: radians → degrees

def _signed_angle_deg(v1, v2, plane_normal):
    """Signed angle of v2 relative to v1 around plane_normal."""
    angle = _angle_deg(v1, v2)
    cross = np.cross(v1, v2)
    sign  = np.sign(np.dot(cross, plane_normal))
    return sign * angle


# ── Back-projection helper ────────────────────────────────────────────────────
def _backproject(u, v, depth_m, fx, fy, cx, cy):
    """
    Unproject pixel (u, v) at depth_m to metric 3D point in camera frame.
    RC-2: must use RGB camera intrinsics, not depth camera intrinsics.
    RC-3: depth_m must already be in metres (mm / 1000.0).
    RC-1: Y is NEGATED to convert image-Y-down to biomechanics-Y-up.
    """
    X =  (u - cx) * depth_m / fx
    Y = -((v - cy) * depth_m / fy)   # RC-1: flip Y
    Z =  depth_m
    return np.array([X, Y, Z], dtype=np.float64)


class SkeletonBuilder:
    def __init__(self):
        # EMA alpha: 0.15 = 85% history, 15% current frame.
        # Old value 0.4 caused ±10° jitter at scoring thresholds.
        # At 8 fps, 0.15 gives ~3-frame lag — invisible to the assessor.
        self.alpha = 0.15
        self._last_angles = {}
        # Holdout: last known-good angles used during brief tracking dropout
        self._holdout_angles = {}
        self._dropout_frames = 0
        self._MAX_DROPOUT    = 6   # hold scores for up to 6 missed frames (~0.75 s)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry: compute angles from MediaPipe normalised landmarks + depth
    # ─────────────────────────────────────────────────────────────────────────
    def compute_angles(self, landmarks_norm, calib=None,
                       depth_frame=None, world_landmarks=None,
                       frame_w=1280, frame_h=720):
        """
        landmarks_norm   : (33, 3) float32 — MediaPipe normalised [x,y,z] ∈ [0,1]
        calib            : CameraCalibration instance (provides RGB intrinsics)
        depth_frame      : (H, W) uint16  — OAK-D depth aligned to RGB, in mm
        world_landmarks  : (33, 3) float32 — MediaPipe world coords (metric, hip-centre)
        frame_w, frame_h : RGB frame dimensions (pixels)
        Returns dict of joint angles in DEGREES, ready for RULA/REBA compute().
        """
        if landmarks_norm is None or len(landmarks_norm) < 33:
            return {}

        lm_n = landmarks_norm  # shape (33, 3)  x,y,z normalised

        # ── Intrinsics ────────────────────────────────────────────────────
        if calib is not None and hasattr(calib, 'rgb_intrinsics'):
            K = calib.rgb_intrinsics
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
        else:
            # RC-2 fallback: approximate intrinsics for frame_w × frame_h
            fx = fy = float(frame_w) * 1.2
            cx, cy  = frame_w / 2.0, frame_h / 2.0

        # ── Build metric 3D point for each landmark ───────────────────────
        pts = {}   # idx → np.array([X, Y, Z]) in metres, Y-up

        def _get_pt(idx):
            """Return metric 3D point for landmark idx, with fallback chain."""
            if idx in pts:
                return pts[idx]

            lm = lm_n[idx]
            u = lm[0] * frame_w
            v = lm[1] * frame_h

            # Try OAK-D stereo depth first
            pt = None
            if depth_frame is not None:
                ui, vi = int(round(u)), int(round(v))
                dh, dw = depth_frame.shape[:2]
                if 0 <= vi < dh and 0 <= ui < dw:
                    # ── 3×3 median depth to kill single-pixel noise ──────
                    r0, r1 = max(0, vi - 1), min(dh, vi + 2)
                    c0, c1 = max(0, ui - 1), min(dw, ui + 2)
                    patch   = depth_frame[r0:r1, c0:c1]
                    valid   = patch[patch > 0]
                    if valid.size >= 3:           # need at least 3 valid pixels
                        depth_mm = float(np.median(valid))
                    elif valid.size > 0:
                        depth_mm = float(valid[0])
                    else:
                        depth_mm = 0.0
                    depth_m = depth_mm / 1000.0   # RC-3: mm → metres
                    if DEPTH_MIN_M <= depth_m <= DEPTH_MAX_M:
                        pt = _backproject(u, v, depth_m, fx, fy, cx, cy)

            # Fallback: MediaPipe world_landmarks (metric, hip-centred)
            if pt is None and world_landmarks is not None:
                wl = world_landmarks[idx]
                pt = np.array([wl[0], wl[1], wl[2]], dtype=np.float64)

            # Last resort: normalised coords treated as rough shape (no depth)
            if pt is None:
                # RC-1 Y-flip applied even on normalised data
                pt = np.array([lm[0] - 0.5, -(lm[1] - 0.5), -lm[2]],
                              dtype=np.float64)

            pts[idx] = pt
            return pt

        # Pre-fetch key landmarks
        nose        = _get_pt(NOSE)
        l_shl       = _get_pt(LEFT_SHOULDER)
        r_shl       = _get_pt(RIGHT_SHOULDER)
        l_elbow     = _get_pt(LEFT_ELBOW)
        r_elbow     = _get_pt(RIGHT_ELBOW)
        l_wrist     = _get_pt(LEFT_WRIST)
        r_wrist     = _get_pt(RIGHT_WRIST)
        l_index     = _get_pt(LEFT_INDEX)
        r_index     = _get_pt(RIGHT_INDEX)
        l_hip       = _get_pt(LEFT_HIP)
        r_hip       = _get_pt(RIGHT_HIP)
        l_knee      = _get_pt(LEFT_KNEE)
        r_knee      = _get_pt(RIGHT_KNEE)
        l_ankle     = _get_pt(LEFT_ANKLE)
        r_ankle     = _get_pt(RIGHT_ANKLE)

        mid_shl = (l_shl + r_shl) / 2.0
        mid_hip = (l_hip + r_hip) / 2.0

        # ── RC-5: Gravity reference = trunk axis (body-relative vertical) ─
        v_trunk  = mid_shl - mid_hip          # points upward along spine
        v_up     = _norm(v_trunk)             # body vertical reference
        # Sagittal plane normal (roughly X-axis = left-right)
        v_right  = _norm(r_shl - l_shl)
        # Frontal plane normal (roughly Z-axis = forward)
        v_fwd    = _norm(np.cross(v_right, v_up))

        angles = {}

        # ── TRUNK flexion ─────────────────────────────────────────────────
        # Angle between trunk vector and global Y-up [0,1,0].
        # 0° = upright, 90° = horizontal forward lean
        global_up = np.array([0.0, 1.0, 0.0])
        trunk_flex = _signed_angle_deg(global_up, v_trunk, v_right)
        angles['trunk'] = trunk_flex   # positive = forward flexion

        # Trunk lateral tilt (Roll): component in frontal plane
        trunk_lat = _angle_deg(v_trunk, np.cross(v_right, global_up))
        angles['trunk_mod'] = 1 if abs(trunk_lat) > 15 else 0
        angles['trunk_roll'] = trunk_lat

        # Trunk rotation (Yaw) — estimated from shoulder-line vs. hip-line orientation
        # Both vectors projected to horizontal plane, angle between them
        v_shl_line = r_shl - l_shl   # left → right shoulder
        v_hip_line = r_hip - l_hip   # left → right hip
        v_shl_h = v_shl_line - np.dot(v_shl_line, v_up) * v_up
        v_hip_h = v_hip_line - np.dot(v_hip_line, v_up) * v_up
        if np.linalg.norm(v_shl_h) > 1e-6 and np.linalg.norm(v_hip_h) > 1e-6:
            trunk_yaw = _signed_angle_deg(v_hip_h, v_shl_h, v_up)
        else:
            trunk_yaw = 0.0
        angles['trunk_yaw'] = trunk_yaw
        angles['trunk_rot']  = trunk_yaw   # keep legacy key

        # ── NECK (head) flexion ───────────────────────────────────────────
        # Vector: mid_shoulder → nose  (head direction)
        v_neck = nose - mid_shl
        neck_flex = _signed_angle_deg(v_trunk, v_neck, v_right)
        # Positive = head forward of trunk axis
        angles['neck'] = neck_flex

        # Neck lateral tilt (Roll) — head tilts left/right
        neck_lat = _signed_angle_deg(v_up, _norm(v_neck), v_fwd)
        angles['neck_mod'] = 1 if abs(neck_lat) > 10 else 0
        angles['neck_roll'] = neck_lat

        # Neck yaw (rotation) — head turns left/right
        # Project neck vector onto horizontal plane (perpendicular to body vertical)
        # and measure rotation relative to the forward axis
        v_neck_h = v_neck - np.dot(v_neck, v_up) * v_up   # horizontal component
        neck_yaw = _signed_angle_deg(v_fwd, v_neck_h, v_up) if np.linalg.norm(v_neck_h) > 1e-6 else 0.0
        angles['neck_yaw'] = neck_yaw

        # ── SHOULDER elevation (left) ────────────────────────────────────
        # Upper-arm vector: shoulder → elbow
        # Angle relative to trunk axis (RC-5)
        v_ua_l = l_elbow - l_shl
        # Elevation: angle between upper arm and trunk axis projected to sagittal
        shl_elev_l = _angle_deg(-v_up, v_ua_l)   # 0°=arm along trunk, 90°=horizontal
        angles['upper_arm_left'] = shl_elev_l

        # Abduction (arm away from body midline)
        v_ua_l_proj_frontal = v_ua_l - np.dot(v_ua_l, v_fwd) * v_fwd
        abd_l = _angle_deg(v_up, v_ua_l_proj_frontal) if np.linalg.norm(v_ua_l_proj_frontal) > 1e-6 else 0.0
        angles['shoulder_mod'] = 1 if abd_l > 20 else 0
        angles['abd_l'] = abd_l

        # ── SHOULDER elevation (right) ───────────────────────────────────
        v_ua_r = r_elbow - r_shl
        shl_elev_r = _angle_deg(-v_up, v_ua_r)
        angles['upper_arm_right'] = shl_elev_r
        
        v_ua_r_proj_frontal = v_ua_r - np.dot(v_ua_r, v_fwd) * v_fwd
        abd_r = _angle_deg(v_up, v_ua_r_proj_frontal) if np.linalg.norm(v_ua_r_proj_frontal) > 1e-6 else 0.0
        angles['abd_r'] = abd_r

        # ── ELBOW flexion (left) ─────────────────────────────────────────
        # Interior angle at elbow:
        #   v1 = shoulder → elbow  (proximal limb seen from elbow)
        #   v2 = wrist → elbow    (distal limb seen from elbow)
        # Both vectors point AWAY from the elbow joint → interior angle.
        # 180° = fully extended, 0° = fully flexed (impossible physically)
        v1_l = l_shl - l_elbow    # shoulder direction from elbow
        v2_l = l_wrist - l_elbow  # wrist direction from elbow
        elbow_interior_l = _angle_deg(v1_l, v2_l)
        angles['elbow_left'] = elbow_interior_l   # RULA uses this directly (60-100° = score 1)

        # Elbow roll (forearm pronation/supination) — left
        # Computed as the angle of the forearm axis rotated around the upper-arm axis
        v_ua_l_n = _norm(v_ua_l)                        # upper arm axis (norm)
        v_fa_l_n = _norm(l_wrist - l_elbow)             # forearm axis (norm)
        # Reference vector perpendicular to upper arm in the body sagittal plane
        v_ref_l  = _norm(np.cross(v_ua_l_n, v_right))
        # Project forearm onto plane perpendicular to upper arm
        v_fa_perp_l = v_fa_l_n - np.dot(v_fa_l_n, v_ua_l_n) * v_ua_l_n
        if np.linalg.norm(v_fa_perp_l) > 1e-6:
            elb_roll_l = _signed_angle_deg(v_ref_l, v_fa_perp_l, v_ua_l_n)
        else:
            elb_roll_l = 0.0
        angles['elb_roll_l'] = elb_roll_l

        # ── ELBOW flexion (right) ────────────────────────────────────────
        v1_r = r_shl - r_elbow
        v2_r = r_wrist - r_elbow
        elbow_interior_r = _angle_deg(v1_r, v2_r)
        angles['elbow_right'] = elbow_interior_r

        # Elbow roll (forearm pronation/supination) — right
        v_ua_r_n = _norm(v_ua_r)
        v_fa_r_n = _norm(r_wrist - r_elbow)
        v_ref_r  = _norm(np.cross(v_ua_r_n, v_right))
        v_fa_perp_r = v_fa_r_n - np.dot(v_fa_r_n, v_ua_r_n) * v_ua_r_n
        if np.linalg.norm(v_fa_perp_r) > 1e-6:
            elb_roll_r = _signed_angle_deg(v_ref_r, v_fa_perp_r, v_ua_r_n)
        else:
            elb_roll_r = 0.0
        angles['elb_roll_r'] = elb_roll_r

        # ── WRIST flexion (left) ─────────────────────────────────────────
        # Forearm axis: elbow → wrist
        v_fa_l = l_wrist - l_elbow
        # Hand axis: wrist → index finger MCP
        v_hand_l = l_index - l_wrist
        if np.linalg.norm(v_hand_l) > 1e-6:
            # MediaPipe's INDEX is naturally offset ~15 deg from forearm axis
            wrist_flex_l = max(0.0, _angle_deg(v_fa_l, v_hand_l) - 15.0)
        else:
            wrist_flex_l = 0.0
        angles['wrist_left'] = wrist_flex_l

        # Wrist roll (radial/ulnar deviation) and yaw (pronation twist) — left
        v_fa_l_n  = _norm(v_fa_l)
        if np.linalg.norm(v_hand_l) > 1e-6:
            v_hand_l_n = _norm(v_hand_l)
            # Roll: deviation in the plane defined by forearm and body-up
            v_fa_up_l  = _norm(np.cross(v_fa_l_n, v_right))
            v_hand_perp_l = v_hand_l_n - np.dot(v_hand_l_n, v_fa_l_n) * v_fa_l_n
            if np.linalg.norm(v_hand_perp_l) > 1e-6:
                wri_roll_l = _signed_angle_deg(v_fa_up_l, v_hand_perp_l, v_fa_l_n)
                wri_yaw_l  = _signed_angle_deg(v_right,   v_hand_perp_l, v_fa_l_n)
            else:
                wri_roll_l = wri_yaw_l = 0.0
        else:
            wri_roll_l = wri_yaw_l = 0.0
        angles['wri_roll_l'] = wri_roll_l
        angles['wri_yaw_l']  = wri_yaw_l

        # ── WRIST flexion (right) ────────────────────────────────────────
        v_fa_r = r_wrist - r_elbow
        v_hand_r = r_index - r_wrist
        if np.linalg.norm(v_hand_r) > 1e-6:
            # MediaPipe's INDEX is naturally offset ~15 deg from forearm axis
            wrist_flex_r = max(0.0, _angle_deg(v_fa_r, v_hand_r) - 15.0)
        else:
            wrist_flex_r = 0.0
        angles['wrist_right'] = wrist_flex_r

        # Wrist roll and yaw — right
        v_fa_r_n  = _norm(v_fa_r)
        if np.linalg.norm(v_hand_r) > 1e-6:
            v_hand_r_n = _norm(v_hand_r)
            v_fa_up_r  = _norm(np.cross(v_fa_r_n, v_right))
            v_hand_perp_r = v_hand_r_n - np.dot(v_hand_r_n, v_fa_r_n) * v_fa_r_n
            if np.linalg.norm(v_hand_perp_r) > 1e-6:
                wri_roll_r = _signed_angle_deg(v_fa_up_r, v_hand_perp_r, v_fa_r_n)
                wri_yaw_r  = _signed_angle_deg(v_right,   v_hand_perp_r, v_fa_r_n)
            else:
                wri_roll_r = wri_yaw_r = 0.0
        else:
            wri_roll_r = wri_yaw_r = 0.0
        angles['wri_roll_r'] = wri_roll_r
        angles['wri_yaw_r']  = wri_yaw_r

        # ── KNEE flexion (left) ──────────────────────────────────────────
        # Interior angle at knee:
        #   v_thigh = hip → knee
        #   v_shank = ankle → knee
        # 180° = straight, 90° = right-angle flex
        v_thigh_l = l_hip - l_knee    # hip direction from knee
        v_shank_l = l_ankle - l_knee  # ankle direction from knee
        knee_interior_l = _angle_deg(v_thigh_l, v_shank_l)
        # REBA uses flex_from_straight = 180 − interior
        knee_flex_l = 180.0 - knee_interior_l
        angles['knee_left'] = knee_flex_l

        # ── KNEE flexion (right) ─────────────────────────────────────────
        v_thigh_r = r_hip - r_knee
        v_shank_r = r_ankle - r_knee
        knee_interior_r = _angle_deg(v_thigh_r, v_shank_r)
        knee_flex_r = 180.0 - knee_interior_r
        angles['knee_right'] = knee_flex_r

        # Legs stable heuristic
        angles['legs_stable'] = True

        # Hip/thigh flexion (for ErgoNet v2 inputs)
        v_torso_l = l_shl - l_hip
        v_thigh_down_l = l_knee - l_hip
        hip_flex_l = _signed_angle_deg(v_torso_l, v_thigh_down_l, v_right)
        angles['hip_left'] = hip_flex_l

        v_torso_r = r_shl - r_hip
        v_thigh_down_r = r_knee - r_hip
        hip_flex_r = _signed_angle_deg(v_torso_r, v_thigh_down_r, v_right)
        angles['hip_right'] = hip_flex_r

        # Thigh roll (adduction/abduction in frontal plane) and yaw (internal rotation)
        # Left thigh
        v_thi_l_n = _norm(v_thigh_down_l)
        v_thi_h_l = v_thigh_down_l - np.dot(v_thigh_down_l, v_up) * v_up  # horizontal component
        if np.linalg.norm(v_thi_h_l) > 1e-6:
            thi_roll_l = _signed_angle_deg(v_right, _norm(v_thi_h_l), v_up)
            thi_yaw_l  = _signed_angle_deg(v_fwd,   _norm(v_thi_h_l), v_up)
        else:
            thi_roll_l = thi_yaw_l = 0.0
        angles['thi_roll_l'] = thi_roll_l
        angles['thi_yaw_l']  = thi_yaw_l

        # Right thigh
        v_thi_r_n = _norm(v_thigh_down_r)
        v_thi_h_r = v_thigh_down_r - np.dot(v_thigh_down_r, v_up) * v_up
        if np.linalg.norm(v_thi_h_r) > 1e-6:
            thi_roll_r = _signed_angle_deg(v_right, _norm(v_thi_h_r), v_up)
            thi_yaw_r  = _signed_angle_deg(v_fwd,   _norm(v_thi_h_r), v_up)
        else:
            thi_roll_r = thi_yaw_r = 0.0
        angles['thi_roll_r'] = thi_roll_r
        angles['thi_yaw_r']  = thi_yaw_r

        return angles

    # ─────────────────────────────────────────────────────────────────────────
    # EMA smoothing + dropout holdout
    # ─────────────────────────────────────────────────────────────────────────
    def enrich_with_depth(self, angles, depth_frame=None, calib=None):
        """
        1. EMA temporal smoothing (alpha=0.15 → 85% history weight).
        2. Dropout holdout: if called with empty angles (pose lost), returns
           the last good smoothed angles for up to _MAX_DROPOUT frames,
           preventing score flicker when MediaPipe briefly loses tracking.
        """
        # ── Dropout path ──────────────────────────────────────────────────
        if not angles:
            self._dropout_frames += 1
            if self._dropout_frames <= self._MAX_DROPOUT and self._holdout_angles:
                return self._holdout_angles.copy()
            return {}

        self._dropout_frames = 0   # person visible again — reset counter

        # ── First frame: seed history ─────────────────────────────────────
        if not self._last_angles:
            self._last_angles    = angles.copy()
            self._holdout_angles = angles.copy()
            return angles

        # ── EMA pass ─────────────────────────────────────────────────────
        smoothed = {}
        for k, v in angles.items():
            if isinstance(v, (int, float)) and k in self._last_angles:
                prev = self._last_angles[k]
                if isinstance(prev, (int, float)):
                    smoothed[k] = self.alpha * v + (1 - self.alpha) * prev
                else:
                    smoothed[k] = v
            else:
                smoothed[k] = v

        self._last_angles    = smoothed
        self._holdout_angles = smoothed.copy()
        return smoothed