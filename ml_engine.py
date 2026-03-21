import os
import pickle
from typing import Tuple, Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import IsolationForest

import config 

class AnomalyEngine:
    """
    Enterprise Anomaly Detection Engine for Satellite Telemetry.

    This engine utilizes unsupervised learning (IsolationForest) to detect
    anomalous patterns in real-time satellite data. It tightly integrates
    with MLflow for experiment tracking and model lineage.
    """
    
    def __init__(self) -> None:
        """Initializes the AnomalyEngine and enforces MLflow tracking configuration."""
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        self._setup_exp()
        self.model: Any = None

    def _setup_exp(self) -> None:
        """
        Idempotently configures the MLflow experiment namespace.
        Creates the experiment if it does not already exist.
        """
        if not mlflow.get_experiment_by_name(config.EXPERIMENT_NAME):
            mlflow.create_experiment(config.EXPERIMENT_NAME)
        mlflow.set_experiment(config.EXPERIMENT_NAME)

    def train(self, X: Any) -> IsolationForest:
        """
        Trains the IsolationForest model on the provided telemetry features and logs artifacts.

        Args:
            X (Any): The feature matrix (e.g., pandas DataFrame or numpy array) for training.

        Returns:
            IsolationForest: The trained Scikit-learn model instance.
        """
        with mlflow.start_run(run_name="Satellite_Retrain"):
            self.model = IsolationForest(
                n_estimators=config.N_ESTIMATORS, 
                contamination=config.CONTAMINATION
            )
            self.model.fit(X)
            
            # Log model architecture and metric lineage to MLflow
            mlflow.sklearn.log_model(self.model, "model")
            
            # Persist model locally as a fallback
            os.makedirs("models", exist_ok=True)
            with open(config.MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
                
        return self.model

    def predict(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes inference on incoming telemetry data to detect anomalies.

        Args:
            X (Any): The incoming feature matrix to evaluate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Array of predictions (1 for nominal, -1 for anomaly).
                - Array of anomaly scores (decision function values).
        """
        if not self.model and os.path.exists(config.MODEL_PATH):
            with open(config.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
                
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        return predictions, scores