# ai/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ErgoNet(nn.Module):
    """
    Advanced Multi-Output Neural Network for Ergonomic Assessment.
    Inputs: 33 MediaPipe Landmarks (x, y, z) = 99 features.
    Outputs: 100+ parameters including Joint Angles, RULA/REBA sub-scores, 
             and Anomaly predictions.
    """
    def __init__(self, input_dim=99, hidden_dim=512):
        super(ErgoNet, self).__init__()
        
        # Shared Encoder (Extracts postural features)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        
        # Head 1: Joint Angles (Regression) - ~20 outputs
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 24) # Flexion, Lateral, Rotation for all joints
        )
        
        # Head 2: RULA/REBA Sub-scores (Classification/Regression) - ~60 outputs
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 80) # Breakdown of all RULA/REBA tables
        )
        
        # Head 3: Anomaly Detection (Probability) - ~10 outputs
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid() # Probability of anomaly per body segment
        )

    def forward(self, x):
        # x shape: [batch, 99]
        features = self.encoder(x)
        
        angles = self.angle_head(features)
        scores = self.score_head(features)
        anomalies = self.anomaly_head(features)
        
        return {
            'angles': angles,
            'scores': scores,
            'anomalies': anomalies
        }

class ErgoForecaster(nn.Module):
    """
    LSTM-based Time-Series Forecaster for 10-day risk prediction.
    Inputs: Sequence of daily average risk scores.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(ErgoForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 10) # Predict next 10 days

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last hidden state
        prediction = self.fc(out)
        return prediction
