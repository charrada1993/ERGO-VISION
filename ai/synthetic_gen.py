# ai/synthetic_gen.py
import numpy as np
import pandas as pd
import os

def generate_ergonomic_dataset(num_samples=10000):
    print(f"[AI] Generating {num_samples} synthetic ergonomic samples...")
    
    data = []
    for _ in range(num_samples):
        # 1. Randomize Joint Angles (within anatomical limits)
        neck_flex = np.random.uniform(-10, 60)
        trunk_flex = np.random.uniform(-10, 90)
        ua_flex_l = np.random.uniform(-20, 150)
        fa_flex_l = np.random.uniform(0, 140)
        
        # 2. Simulate 3D Landmarks based on these angles
        # (Simplified kinematic chain for synthetic bootstrap)
        # Assuming origin at Hip
        lms = np.zeros((33, 3))
        
        # Hip (origin)
        lms[23] = [0.1, 0, 0] # Left Hip
        lms[24] = [-0.1, 0, 0] # Right Hip
        
        # Trunk + Shoulder
        trunk_rad = np.radians(trunk_flex)
        shl_y = 0.5 * np.cos(trunk_rad)
        shl_z = 0.5 * np.sin(trunk_rad)
        lms[11] = [0.2, shl_y, shl_z] # Left Shoulder
        
        # Neck + Head
        neck_rad = np.radians(neck_flex)
        head_y = shl_y + 0.2 * np.cos(neck_rad)
        head_z = shl_z + 0.2 * np.sin(neck_rad)
        lms[0] = [0, head_y, head_z] # Nose
        
        # Upper Arm
        ua_rad = np.radians(ua_flex_l)
        el_y = shl_y + 0.3 * np.cos(ua_rad)
        el_z = shl_z + 0.3 * np.sin(ua_rad)
        lms[13] = [0.2, el_y, el_z] # Left Elbow
        
        # 3. Add Noise (simulate camera/sensor jitter)
        lms += np.random.normal(0, 0.01, lms.shape)
        
        # 4. Flatten for CSV
        row = {}
        for i in range(33):
            row[f'lm_{i}_x'] = lms[i][0]
            row[f'lm_{i}_y'] = lms[i][1]
            row[f'lm_{i}_z'] = lms[i][2]
            
        # Ground Truth (Labels)
        row['Neck_Flexion_deg'] = neck_flex
        row['Trunk_Flexion_deg'] = trunk_flex
        row['L_Shoulder_Flexion_deg'] = ua_flex_l
        
        # Calculate RULA (simplified for labels)
        rula_score = 1
        if neck_flex > 20: rula_score += 2
        if trunk_flex > 20: rula_score += 2
        if ua_flex_l > 45: rula_score += 2
        
        row['RULA_score'] = rula_score
        row['REBA_score'] = rula_score * 1.5 # Placeholder for REBA logic
        
        # Add the 100+ parameters (placeholders for training)
        for i in range(24): row[f'angle_{i}'] = np.random.uniform(0, 1)
        for i in range(80): row[f'score_{i}'] = np.random.uniform(0, 1)
        
        data.append(row)
        
    df = pd.DataFrame(data)
    os.makedirs("ai/data", exist_ok=True)
    df.to_csv("ai/data/dataset.csv", index=False)
    print("[AI] Synthetic dataset saved to ai/data/dataset.csv")

if __name__ == "__main__":
    generate_ergonomic_dataset()
