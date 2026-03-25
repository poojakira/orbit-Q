import numpy as np
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyTorch could not be loaded: {e}")
    TORCH_AVAILABLE = False
    
    # Mock classes for import safety
    class nn:
        Module = object


class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 8):
        super(PyTorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # assuming normalized input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderAnomalyDetector:
    """Wrapper to make PyTorch AE behave like a scikit-learn anomaly detector."""

    def __init__(self, input_dim: int = 3, epochs: int = 10, lr: float = 1e-3, threshold_percentile: float = 95.0):
        if not TORCH_AVAILABLE:
            self.model = None
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PyTorchAutoencoder(input_dim=input_dim).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.threshold = 0.0

    def fit(self, X: np.ndarray):
        if not TORCH_AVAILABLE: return self
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()

        # Calculate reconstruction error to set the threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            self.threshold = float(np.percentile(mse, self.threshold_percentile))
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE: return np.ones(len(X))
        
        scores = self.decision_function(X)
        preds = np.where(scores > self.threshold, -1, 1)
        return preds

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE: return np.zeros(len(X))
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        return mse
