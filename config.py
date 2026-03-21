import os

# Infrastructure
DB_URL = os.getenv("FIREBASE_DB_URL", "https://YOUR-NEW-PROJECT-ID.firebasedatabase.app/") #
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT_PATH", "service_account.json") #

# ML Governance
MLFLOW_URI = "sqlite:///mlflow.db" #
EXPERIMENT_NAME = "CubeSat_3D_Anomaly_Engine" #
MODEL_PATH = "models/isolation_forest_latest.pkl" #

# Hyperparameters
CONTAMINATION = 0.05 #
N_ESTIMATORS = 200 #
ROLLING_WINDOW = 5 #
