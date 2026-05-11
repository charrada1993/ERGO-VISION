# ai/operation/inference.py
import numpy as np
import pickle
import os

class ErgoAIPlus:
    """
    The Operational AI Engine for ERGO-VISION.
    This class handles real-time inference using the trained Numpy model.
    """
    def __init__(self, model_path="ai/models/ergo_net_numpy.pkl"):
        self.model_path = model_path
        self.model_data = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[AI] ERROR: Operational model not found at {self.model_path}")
            return
        
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        print(f"[AI] Operational AI Engine loaded successfully.")

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, landmarks_3d):
        """
        Input: 33 x 3 list/array of landmarks.
        Output: Dictionary of 100+ ergonomic parameters.
        """
        if self.model_data is None:
            return None

        # 1. Prepare Input (Flatten and Normalize)
        x = np.array(landmarks_3d).flatten().reshape(1, -1)
        x_norm = (x - self.model_data['X_mean']) / self.model_data['X_std']

        # 2. Forward Pass (Inference)
        z1 = np.dot(x_norm, self.model_data['W1']) + self.model_data['b1']
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.model_data['W2']) + self.model_data['b2']
        
        # 3. De-normalize Output
        y_norm = z2
        y = y_norm * self.model_data['y_mean'] + self.model_data['y_mean'] # Corrected denorm
        y = y_norm * self.model_data['y_std'] + self.model_data['y_mean']
        
        # 4. Map to Result Dictionary
        res_list = y.flatten()
        target_cols = self.model_data['target_cols']
        results = {target_cols[i]: float(res_list[i]) for i in range(len(target_cols))}
        
        return results
