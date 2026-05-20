# data/logger.py
import csv
import time
import os
from datetime import datetime
from config import Config

class DataLogger:
    CONDITIONS = [
        "Normal",
        "Carpal Tunnel", "Cervical Disc Risk", "Cervicalgia", "De Quervain",
        "Elbow Epicondylitis", "Elbow Strain", "Frozen Shoulder", "Hip Bursitis",
        "Hip Flexor Strain", "Low Back Pain", "Lumbar Disc Risk",
        "Postural Kyphosis", "Rotator Cuff Tendinitis", "Shoulder Bursitis",
        "Shoulder Impingement", "Tech Neck", "Wrist Tendinitis"
    ]
    SEVERITIES = ["Healthy", "Low Risk", "Moderate", "High Risk", "Critical"]
    LOCATIONS = [
        "None", "Wrist", "Elbow", "Shoulder", "Neck", "Upper Back",
        "Lower Back", "Hip", "Knee", "Full Body"
    ]

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
        # Detailed columns including both sides + AI Metrics
        self.writer.writerow([
            "timestamp", "frame_id", 
            "neck_deg", "trunk_deg", 
            "ua_left_deg", "ua_right_deg",
            "el_left_deg", "el_right_deg", 
            "wr_left_deg", "wr_right_deg",
            "RULA_score", "REBA_score",
            "risk_prediction", "anomalies",
            "ai_risk_score", "ai_severity_code", "ai_severity",
            "ai_location_code", "ai_location", 
            "ai_condition_code", "ai_condition"
        ])
        self.start_time = time.time()
        self.sample_count = 0
        return filename

    def log(self, angles, rula_result, reba_result, anomalies, ai_results=None):
        if self.writer is None or self.file is None:
            return
        elapsed = time.time() - self.start_time
        self.sample_count += 1
        
        # Extract AI results if available
        ai_res = ai_results if ai_results else {}
        
        sev_code = int(ai_res.get('severity_code', 0))
        loc_code = int(ai_res.get('location_code', 0))
        cond_code = int(ai_res.get('condition_code', 0))
        
        sev_str = self.SEVERITIES[sev_code] if 0 <= sev_code < len(self.SEVERITIES) else "Unknown"
        loc_str = self.LOCATIONS[loc_code] if 0 <= loc_code < len(self.LOCATIONS) else "Unknown"
        cond_str = self.CONDITIONS[cond_code] if 0 <= cond_code < len(self.CONDITIONS) else "Unknown"
        
        row = [
            round(elapsed, 3),
            self.sample_count,
            round(angles.get('neck', 0), 2),
            round(angles.get('trunk', 0), 2),
            round(angles.get('upper_arm_left', 0), 2),
            round(angles.get('upper_arm_right', 0), 2),
            round(angles.get('elbow_left', 0), 2),
            round(angles.get('elbow_right', 0), 2),
            round(angles.get('wrist_left', 0), 2),
            round(angles.get('wrist_right', 0), 2),
            rula_result.get('RULA_score', 0),
            reba_result.get('REBA_score', 0),
            rula_result.get('risk_level', 'Low'),
            "; ".join(anomalies) if anomalies else "None",
            round(ai_res.get('risk_score', 0.0), 2),
            sev_code, sev_str,
            loc_code, loc_str,
            cond_code, cond_str
        ]
        self.writer.writerow(row)
        self.file.flush()

    def end_session(self):
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None