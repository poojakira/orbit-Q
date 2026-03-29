import os
import warnings

# Infrastructure
# Set FIREBASE_DB_URL in your environment before running.
# Example: export FIREBASE_DB_URL="https://<your-project>.firebasedatabase.app/"
_db_url = os.getenv("FIREBASE_DB_URL", "")
if not _db_url:
    warnings.warn(
        "FIREBASE_DB_URL environment variable is not set. "
        "Firebase-dependent features will be unavailable.",
        RuntimeWarning,
        stacklevel=1,
    )
DB_URL: str = _db_url

SERVICE_ACCOUNT: str = os.getenv("SERVICE_ACCOUNT_PATH", "service_account.json")

# ML Governance
MLFLOW_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "CubeSat_3D_Anomaly_Engine"
)
MODEL_PATH: str = os.getenv(
    "MODEL_PATH", "models/orbit_q_ensemble_latest.pkl"
)

# Hyperparameters
CONTAMINATION: float = float(os.getenv("CONTAMINATION", "0.05"))
N_ESTIMATORS: int = int(os.getenv("N_ESTIMATORS", "200"))
ROLLING_WINDOW: int = int(os.getenv("ROLLING_WINDOW", "5"))
