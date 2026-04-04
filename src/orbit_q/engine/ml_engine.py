import logging
import os
import pickle
from typing import Any, Tuple

import mlflow
import mlflow.sklearn
import numpy as np

# Try importing RAPIDS cuML for GPU acceleration
try:
    from cuml.ensemble import IsolationForest

    gpu_enabled = True
except ImportError:
    from sklearn.ensemble import IsolationForest

    gpu_enabled = False

from orbit_q import config
from orbit_q.engine.kernels.anomaly_fusion import fuse_scores
from orbit_q.engine.models.autoencoder import AutoencoderAnomalyDetector
from orbit_q.engine.models.lstm_detector import LSTMTemporalDetector

log = logging.getLogger(__name__)


class AnomalyEngine:
    """
    Enterprise Anomaly Detection Engine for Satellite Telemetry.

    Three-model ensemble:
    1. IsolationForest - global outlier detection (GPU via cuML if available)
    2. PyTorch Autoencoder - reconstruction-error anomaly detection
    3. LSTM Temporal Detector - sequence-level temporal pattern anomaly detection

    Scores are fused via a CUDA Triton kernel (falls back to NumPy on CPU).
    """

    def __init__(self) -> None:
        """Initialises the AnomalyEngine with MLflow tracking."""
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        self._setup_exp()
        self.iso_model: Any = None
        self.ae_model: Any = None
        self.lstm_model: Any = None

    def _setup_exp(self) -> None:
        if not mlflow.get_experiment_by_name(config.EXPERIMENT_NAME):
            mlflow.create_experiment(config.EXPERIMENT_NAME)
        mlflow.set_experiment(config.EXPERIMENT_NAME)

    def train(self, X: Any) -> Tuple[Any, Any]:
        """Trains the three-model ensemble and logs artifacts to MLflow."""
        with mlflow.start_run(run_name="Satellite_Ensemble_Retrain"):
            mlflow.log_param("gpu_enabled", gpu_enabled)
            # 1. Train Isolation Forest
            self.iso_model = IsolationForest(
                n_estimators=config.N_ESTIMATORS,
                contamination=config.CONTAMINATION,
            )
            self.iso_model.fit(X)
            # 2. Train Autoencoder
            try:
                input_dim = X.shape[1] if hasattr(X, "shape") else len(X[0])
            except Exception:
                input_dim = 5
            self.ae_model = AutoencoderAnomalyDetector(input_dim=input_dim, epochs=20)
            self.ae_model.fit(X)
            # 3. Train LSTM Temporal Detector
            self.lstm_model = LSTMTemporalDetector(input_dim=input_dim, epochs=15)
            self.lstm_model.fit(X)
            # Log and REGISTER the models for version control
            # We log the sub-models individually to the registry
            mlflow.sklearn.log_model(
                self.iso_model, 
                "iso_model",
                registered_model_name="Orbit-Q-IsolationForest"
            )
            mlflow.log_param("ensemble_models", "IsolationForest+Autoencoder+LSTM")
            os.makedirs("models", exist_ok=True)
            with open(config.MODEL_PATH, "wb") as f:
                pickle.dump(
                    {"iso": self.iso_model, "ae": self.ae_model, "lstm": self.lstm_model},
                    f,
                )
        return self.iso_model, self.ae_model

    def predict(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Run the three-model ensemble and fuse scores via Triton kernel."""
        # Load from disk if models not in memory
        if (self.iso_model is None or self.ae_model is None) and os.path.exists(config.MODEL_PATH):
            with open(config.MODEL_PATH, "rb") as f:
                models = pickle.load(f)
            if isinstance(models, dict):
                self.iso_model = models.get("iso", models)
                self.ae_model = models.get("ae", None)
                self.lstm_model = models.get("lstm", None)
            else:
                self.iso_model = models
                self.ae_model = None
                self.lstm_model = None

        if self.iso_model is None:
            raise RuntimeError("Model not trained. Call engine.train(X) first.")

        # 1. IsolationForest scores
        iso_scores = self.iso_model.decision_function(X).astype(np.float32)
        # 2. Autoencoder scores
        if self.ae_model is not None:
            ae_scores = self.ae_model.decision_function(X).astype(np.float32)
        else:
            ae_scores = np.zeros(len(iso_scores), dtype=np.float32)
        # 3. LSTM scores (pad/trim to match length)
        if self.lstm_model is not None:
            lstm_raw = self.lstm_model.decision_function(X).astype(np.float32)
            n = len(iso_scores)
            if len(lstm_raw) >= n:
                lstm_scores = lstm_raw[:n]
            else:
                lstm_scores = np.pad(lstm_raw, (0, n - len(lstm_raw)), mode="edge")
        else:
            lstm_scores = np.zeros(len(iso_scores), dtype=np.float32)
        # 4. Fuse IsolationForest + Autoencoder via Triton kernel (or NumPy fallback)
        fused_iso_ae = fuse_scores(iso_scores, ae_scores, iso_weight=0.6)
        # 5. Incorporate LSTM: weighted average with normalised LSTM score
        lstm_norm = lstm_scores / (lstm_scores.max() + 1e-9)
        ensemble_scores = 0.7 * fused_iso_ae + 0.3 * (1.0 - np.clip(lstm_norm, 0.0, 1.0))
        # 6. Classify: fused score < 0.5 means anomaly (-1), else nominal (+1)
        ensemble_preds = np.where(ensemble_scores < 0.5, -1, 1).astype(int)
        log.debug(
            "Ensemble predict | samples=%d anomalies=%d",
            len(ensemble_preds),
            int((ensemble_preds == -1).sum()),
        )
        return ensemble_preds, ensemble_scores
