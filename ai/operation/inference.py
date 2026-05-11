# ai/operation/inference.py
import pickle
import numpy as np
import os

class NumpyInference:
    """
    Inference wrapper for ErgoNet.
    Supports Version 1 (Landmarks) and Version 2 (Angles).
    """
    def __init__(self, model_path=None):
        # Default search order: v2 then v1
        if model_path is None:
            if os.path.exists("ai/models/ergo_net_v2.pkl"):
                model_path = "ai/models/ergo_net_v2.pkl"
            elif os.path.exists("ai/models/ergo_net_numpy.pkl"):
                model_path = "ai/models/ergo_net_numpy.pkl"
            else:
                raise FileNotFoundError("No ErgoNet model found in ai/models/")
            
        with open(model_path, 'rb') as f:
            self.state = pickle.load(f)
            
        self.version = self.state.get('version', '1.0')
        self.W1 = self.state['W1']
        self.b1 = self.state['b1']
        self.W2 = self.state['W2']
        self.b2 = self.state['b2']
        self.X_mean = self.state['X_mean']
        self.X_std = self.state['X_std']
        self.y_mean = self.state['y_mean']
        self.y_std = self.state['y_std']
        self.target_cols = self.state['target_cols']
        self.input_cols = self.state.get('input_cols', []) # Only for v2
        
        print(f"[AI] Loaded ErgoNet {self.version} with {len(self.target_cols)} outputs.")

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, input_data):
        """
        input_data: 
          - if v1: np.ndarray of landmarks (33, 3) or (99,)
          - if v2: dict of angles or list/array of angles
        """
        # 1. Prepare input vector
        if self.version == '2.0':
            if isinstance(input_data, dict):
                # Map SkeletonBuilder keys to Model V2 Training keys
                mapping = {
                    'Neck_Flexion_deg':       input_data.get('neck', 0),
                    'Trunk_Flexion_deg':      input_data.get('trunk', 0),
                    'L_Shoulder_Flexion_deg': input_data.get('upper_arm_left', 0),
                    'R_Shoulder_Flexion_deg': input_data.get('upper_arm_right', 0),
                    'L_Elbow_Flexion_deg':    input_data.get('elbow_left', 0),
                    'R_Elbow_Flexion_deg':    input_data.get('elbow_right', 0),
                    'L_Wrist_Deviation_deg':  input_data.get('wrist_left', 0),
                    'R_Wrist_Deviation_deg':  input_data.get('wrist_right', 0),
                    'L_Hip_Flexion_deg':      0, # Placeholder
                    'R_Hip_Flexion_deg':      0, # Placeholder
                    'L_Knee_Flexion_deg':     0, # Placeholder
                    'R_Knee_Flexion_deg':     0  # Placeholder
                }
                x = np.array([mapping.get(col, 0) for col in self.input_cols]).reshape(1, -1)
            else:
                x = np.array(input_data).reshape(1, -1)
        else:
            # v1: Landmarks
            x = np.array(input_data).flatten().reshape(1, -1)
            
        # 2. Normalize
        x_norm = (x - self.X_mean) / self.X_std
        
        # 3. Forward pass
        z1 = np.dot(x_norm, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        
        # 4. Denormalize output
        y_orig = z2 * self.y_std + self.y_mean
        
        # 5. Map to dictionary
        results = {}
        for i, col in enumerate(self.target_cols):
            results[col] = float(y_orig[0, i])
            
        return results

if __name__ == "__main__":
    # Test script
    try:
        inf = NumpyInference()
        if inf.version == '2.0':
            dummy = {col: 10.0 for col in inf.input_cols}
        else:
            dummy = np.random.randn(99)
        res = inf.predict(dummy)
        print(f"Prediction (v{inf.version}):", res)
    except Exception as e:
        print(f"Error: {e}")
