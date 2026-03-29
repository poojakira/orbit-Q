"""
Temporal Anomaly Detection - LSTM Model
Uses a sequence-to-sequence LSTM to learn nominal telemetry patterns over time.
Reconstruction error on held-out sequences triggers anomaly labels.
"""
import numpy as np
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyTorch unavailable ({e}) - LSTM disabled")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class LSTMAutoencoder(nn.Module):  # type: ignore[misc]
        """Sequence-to-sequence LSTM for temporal anomaly detection."""

        def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_layers: int = 2):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

        def forward(self, x):  # x: (batch, seq_len, features)
            _, (h, c) = self.encoder(x)
            batch_size, seq_len = x.size(0), x.size(1)
            decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder(decoder_input)
            return out

else:

    class LSTMAutoencoder(object):  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            pass


class LSTMTemporalDetector:
    """
    Wraps LSTMAutoencoder and exposes a scikit-learn-compatible interface.
    Input shape: (n_sequences, seq_len, n_features).
    """

    def __init__(
        self,
        input_dim: int = 5,
        seq_len: int = 20,
        hidden_dim: int = 32,
        epochs: int = 15,
        lr: float = 1e-3,
        threshold_percentile: float = 95.0,
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.threshold = 0.0
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = LSTMAutoencoder(input_dim, hidden_dim).to(self.device)
        else:
            self.device = None
            self.model = None

    # --- Helpers ---
    def _to_sequences(self, X: np.ndarray) -> np.ndarray:
        """Slide a window of seq_len over X to get (n_windows, seq_len, features)."""
        seqs = []
        for i in range(len(X) - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])
        return np.array(seqs, dtype=np.float32)

    # --- Public API ---
    def fit(self, X: np.ndarray) -> "LSTMTemporalDetector":
        if not TORCH_AVAILABLE:
            return self
        seqs = self._to_sequences(X)
        X_t = torch.tensor(seqs).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(X_t)
            loss = criterion(recon, X_t)
            loss.backward()
            optimizer.step()
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_t)
            err = torch.mean((X_t - recon) ** 2, dim=(1, 2)).cpu().numpy()
        self.threshold = float(np.percentile(err, self.threshold_percentile))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return per-timestep reconstruction error (higher = more anomalous)."""
        if not TORCH_AVAILABLE:
            return np.zeros(len(X))
        seqs = self._to_sequences(X)
        X_t = torch.tensor(seqs).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_t)
            err = torch.mean((X_t - recon) ** 2, dim=(1, 2)).cpu().numpy()
        pad = np.full(self.seq_len - 1, err[0])
        return np.concatenate([pad, err])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return -1 for anomaly, +1 for nominal."""
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, -1, 1).astype(int)
