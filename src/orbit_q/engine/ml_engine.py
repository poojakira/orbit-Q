import os
import pickle
from typing import Tuple, Any

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
from orbit_q.engine.models.autoencoder import AutoencoderAnomalyDetector

class AnomalyEngine:
    """
    Enterprise Anomaly Detection Engine for Satellite Telemetry.
    Uses an Ensemble of GPU-accelerated IsolationForest and PyTorch Autoencoder.
    """
    
    def __init__(self) -> None:
        """Initializes the AnomalyEngine with MLflow tracking."""
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        self._setup_exp()
        self.iso_model: Any = None
        self.ae_model: Any = None

    def _setup_exp(self) -> None:
        if not mlflow.get_experiment_by_name(config.EXPERIMENT_NAME):
            mlflow.create_experiment(config.EXPERIMENT_NAME)
        mlflow.set_experiment(config.EXPERIMENT_NAME)

    def train(self, X: Any) -> Tuple[Any, Any]:
        """Trains the ensemble and logs artifacts."""
        with mlflow.start_run(run_name="Satellite_Ensemble_Retrain"):
            mlflow.log_param("gpu_enabled", gpu_enabled)
            
            # 1. Train Isolation Forest
            self.iso_model = IsolationForest(
                n_estimators=config.N_ESTIMATORS, 
                contamination=config.CONTAMINATION
            )
            self.iso_model.fit(X)
            
            # 2. Train Autoencoder
            try:
                input_dim = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            except Exception:
                input_dim = 3
                
            self.ae_model = AutoencoderAnomalyDetector(input_dim=input_dim, epochs=20)
            self.ae_model.fit(X)
            
            # Log models
            mlflow.sklearn.log_model(self.iso_model, "iso_model")
            
            os.makedirs("models", exist_ok=True)
            with open(config.MODEL_PATH, "wb") as f:
                pickle.dump({"iso": self.iso_model, "ae": self.ae_model}, f)
                
        return self.iso_model, self.ae_model

    def predict(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        if (not self.iso_model or not self.ae_model) and os.path.exists(config.MODEL_PATH):
            with open(config.MODEL_PATH, "rb") as f:
                models = pickle.load(f)
                if isinstance(models, dict):
                    self.iso_model = models.get('iso', models)
                    self.ae_model = models.get('ae', None)
                else:
                    self.iso_model = models
                    self.ae_model = None
                
        iso_preds = self.iso_model.predict(X)
        iso_scores = self.iso_model.decision_function(X)
        
        if self.ae_model:
            ae_preds = self.ae_model.predict(X)
            ae_scores = self.ae_model.decision_function(X)
            ensemble_preds = np.where((iso_preds == -1) | (ae_preds == -1), -1, 1)
            ensemble_scores = iso_scores
        else:
            ensemble_preds = iso_preds
            ensemble_scores = iso_scores

        return ensemble_preds, ensemble_scores