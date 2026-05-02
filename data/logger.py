# data/logger.py
import csv
import time
import os
from datetime import datetime
from config import Config

class DataLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.start_time = None
        self.session_path = None
        self.sample_count = 0

    def start_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.csv"
        os.makedirs(Config.SESSION_DIR, exist_ok=True)
        self.session_path = os.path.join(Config.SESSION_DIR, filename)
        self.file = open(self.session_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "frame_id", "neck_deg", "trunk_deg", "upper_arm_deg",
                              "elbow_deg", "wrist_deg", "RULA_score", "REBA_score",
                              "risk_prediction", "anomalies"])
        self.start_time = time.time()
        self.sample_count = 0
        return filename

    def log(self, angles, rula_result, reba_result, anomalies):
        if self.writer is None or self.file is None:
            return
        elapsed = time.time() - self.start_time
        self.sample_count += 1
        row = [
            round(elapsed, 3),
            self.sample_count,
            round(angles.get('neck', 0), 2),
            round(angles.get('trunk', 0), 2),
            round(angles.get('upper_arm_left', 0), 2),
            round(angles.get('elbow_left', 0), 2),
            round(angles.get('wrist_left', 0), 2),
            rula_result.get('RULA_score', 0),
            reba_result.get('REBA_score', 0),
            rula_result.get('risk_level', 'Low'),
            "; ".join(anomalies) if anomalies else "None"
        ]
        self.writer.writerow(row)
        self.file.flush()

    def end_session(self):
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None