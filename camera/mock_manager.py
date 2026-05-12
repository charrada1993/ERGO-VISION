# camera/mock_manager.py
import math
import numpy as np
import cv2
import time
import threading
from config import JetsonConfig

class MockCameraManager:
    """
    Simulated camera for ERGO-VISION.
    Generates synthetic RGB frames and depth maps when no OAK-D is connected.
    """
    def __init__(self, pipeline=None, device=None):
        self.running = False
        self.frame_rgb = None
        self.frame_depth = None
        self.frame_disp = None
        self._lock = threading.Lock()
        
        # Dimensions from config
        self.w = JetsonConfig.RGB_WIDTH
        self.h = JetsonConfig.RGB_HEIGHT
        
    def setup(self):
        print("[MockCamera] Virtual pipeline initialized.")
        return True
        
    def start_streams(self):
        self.running = True
        self.thread = threading.Thread(target=self._generator, daemon=True)
        self.thread.start()
        print("[MockCamera] Simulation started (8 FPS).")
        
    def _generator(self):
        while self.running:
            # 1. Create a dark gradient background
            img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            cv2.putText(img, "SIMULATION MODE", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 212, 255), 2)
            cv2.putText(img, time.strftime("%H:%M:%S"), (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            # 2. Draw a "person" (simple circles) that moves slightly
            t = time.time()
            offset = int(math.sin(t) * 20)
            # Head
            cv2.circle(img, (self.w//2 + offset, self.h//3), 30, (255, 255, 255), -1)
            # Torso
            cv2.line(img, (self.w//2 + offset, self.h//3 + 30), 
                     (self.w//2 + offset, self.h//3 + 150), (255, 255, 255), 5)
            
            # 3. Mock depth map
            depth = np.full((self.h, self.w), 1500, dtype=np.uint16) # 1.5m constant depth
            
            with self._lock:
                self.frame_rgb = img
                self.frame_depth = depth
                self.frame_disp = (depth / 10).astype(np.uint8)
                
            time.sleep(0.125) # 8 FPS

    def get_latest_frames(self):
        with self._lock:
            return {
                'timestamp': time.time(),
                'rgb':       self.frame_rgb,
                'depth':     self.frame_depth,
                'disp':      self.frame_disp,
            }

    def get_depth_at_point(self, x, y):
        return 1.5

    def stop(self):
        self.running = False

