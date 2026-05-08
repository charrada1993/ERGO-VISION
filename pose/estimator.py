# pose/estimator.py
# Optimized for Jetson Orin: resize frame to a smaller resolution BEFORE
# passing to MediaPipe. This is the single biggest CPU saving in the pipeline.
# MediaPipe landmarks are normalised 0-1, so they map correctly to any
# original resolution without any coordinate correction.

import mediapipe as mp
import cv2
import numpy as np
from config import JetsonConfig

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,          # Lite model – fastest on ARM
    smooth_landmarks=False,       # Disabled – saves per-frame interpolation
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Target inference resolution: 320×180
# At 416×320 input from camera, a resize to 320×180 is very fast (INTER_LINEAR).
_INF_W = JetsonConfig.POSE_INPUT_WIDTH   # 320
_INF_H = JetsonConfig.POSE_INPUT_HEIGHT  # 180


class PoseEstimator:
    @staticmethod
    def get_landmarks(frame):
        """
        Run MediaPipe Pose on a single BGR frame.
        The frame is first resized to (_INF_W × _INF_H) to reduce CPU load.
        Returned landmarks are normalised [0, 1] and do not need rescaling.
        Returns np.ndarray of shape (33, 3) or None.
        """
        if frame is None:
            return None

        # ── Resize before inference (biggest single CPU saving) ─────────
        small = cv2.resize(frame, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR)

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)

        if results.pose_landmarks:
            landmarks = [
                [lm.x, lm.y, lm.z]
                for lm in results.pose_landmarks.landmark
            ]
            return np.array(landmarks, dtype=np.float32)
        return None