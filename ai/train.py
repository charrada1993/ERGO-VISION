# ai/train.py
import numpy as np
import pandas as pd
import os
import pickle

class NumpyErgoNet:
    """
    High-performance Neural Network implemented in raw Numpy.
    Zero-dependencies, optimized for Jetson ARM cores.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def train(self, X, y, epochs=100, lr=0.01):
        print(f"[AI] Training NumpyErgoNet for {epochs} epochs...")
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            
            # Backward
            error = 2 * (output - y) / X.shape[0]
            dW2 = np.dot(self.a1.T, error)
            db2 = np.sum(error, axis=0, keepdims=True)
            
            d_a1 = np.dot(error, self.W2.T)
            d_z1 = d_a1 * self.relu_deriv(self.z1)
            
            dW1 = np.dot(X.T, d_z1)
            db1 = np.sum(d_z1, axis=0, keepdims=True)
            
            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

def main():
    # 1. Load Data
    # Logic to handle paths correctly whether run from root or ai/ directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    
    if not os.path.exists(data_path):
        print(f"[AI] ERROR: Dataset {data_path} not found. Run download_dataset.py first.")
        return

    df = pd.read_csv(data_path)
    lm_cols = [c for c in df.columns if c.startswith('lm_')]
    # Select all 100+ target columns (angles, scores, etc.)
    target_cols = [c for c in df.columns if not c.startswith('lm_') and c != 'timestamp']
    
    X = df[lm_cols].values
    y = df[target_cols].values
    
    # Normalize X
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std
    
    # Normalize Y (Crucial to prevent NaN loss with 100+ outputs)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
    y_norm = (y - y_mean) / y_std

    # 2. Train
    model = NumpyErgoNet(input_dim=len(lm_cols), hidden_dim=256, output_dim=len(target_cols))
    model.train(X_norm, y_norm, epochs=300, lr=0.001)

    # 3. Save Model + Normalization Stats
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    state = {
        'W1': model.W1, 'b1': model.b1,
        'W2': model.W2, 'b2': model.b2,
        'X_mean': X_mean, 'X_std': X_std,
        'y_mean': y_mean, 'y_std': y_std,
        'target_cols': target_cols
    }
    save_path = os.path.join(models_dir, "ergo_net_numpy.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"[AI] Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
