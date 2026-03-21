"""
Sensor Anomaly Detection Pipeline
Author: Pooja Kiran (MLOps Project)
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import firebase_admin
from firebase_admin import credentials, db
import logging

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# CONFIG - Environment variables replace hardcoded strings
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL", "https://cubesat-5403b-default-rtdb.asia-southeast1.firebasedatabase.app/")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "service_account.json")
MODEL_PATH = "model.pkl"
ROLLING_WINDOW = 5
ANOMALY_CONTAMINATION = 0.05

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    logging.info("Firebase initialized")

def fetch_sensor_data() -> pd.DataFrame:
    # Scalability fix: limit to last 1000 records
    ref = db.reference("/SENSOR_DATA")
    data = ref.order_by_key().limit_to_last(1000).get()
    if not data: return pd.DataFrame()
    df = pd.DataFrame.from_records(list(data.values()))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    # Feature windowing as described in project overview
    for feat in ["rolling_mean", "rolling_std"]:
        df[feat] = df.groupby("face")["distance_cm"].transform(
            lambda x: x.rolling(ROLLING_WINDOW).mean() if "mean" in feat else x.rolling(ROLLING_WINDOW).std()
        )
    return df.dropna()

def run_pipeline():
    init_firebase()
    df = fetch_sensor_data()
    if df.empty: return
    
    df = build_features(df)
    features = df[["distance_cm", "rolling_mean", "rolling_std"]]
    
    # Load or Train Model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
    else:
        model = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
        model.fit(features)
        with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)

    df["prediction"] = model.predict(features)
    anomalies = df[df["prediction"] == -1]
    
    if not anomalies.empty:
        alert_ref = db.reference("/ML_ALERTS")
        for _, row in anomalies.iterrows():
            alert_ref.push({
                "timestamp": row["timestamp"].isoformat(),
                "face": row["face"],
                "score": float(model.decision_function(features)[0]),
                "source": "ml_anomaly_service"
            })
        logging.info(f"Pushed {len(anomalies)} alerts.")

if __name__ == "__main__":
    run_pipeline()
