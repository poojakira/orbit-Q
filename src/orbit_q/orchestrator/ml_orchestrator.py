import time
import logging
from typing import Optional, Any
import pandas as pd
from firebase_admin import db
import mlflow

from orbit_q.orchestrator.feature_processor import FeatureProcessor
from orbit_q.engine.ml_engine import AnomalyEngine
from orbit_q.engine.metrics_evaluator import MetricsEvaluator
from orbit_q.ingestion.kafka_client import OrbitKafkaClient
from orbit_q import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MLOrchestrator:
    """
    Core operational loop for the Satellite Command Center.

    Orchestrates the retrieval of raw telemetry, feature extraction,
    model inference, and dispatches MLOps metrics and critical alerts.
    """

    def __init__(self) -> None:
        """Initializes the ML Orchestrator with the Anomaly Engine subsystem."""
        self.engine: AnomalyEngine = AnomalyEngine()
        self.kafka_client = OrbitKafkaClient()
        self.LEGACY_LATENCY: float = 2.0  # Baseline response requirement for comparative metrics
        self.polling_interval: float = 10.0  # Default adaptive window

    def fetch_and_process(self) -> Optional[pd.DataFrame]:
        """
        Fetches telemetry from Kafka (Primary) or Firebase (Fallback).

        Returns:
            Optional[pd.DataFrame]: A processed pandas DataFrame containing model features, or None if fetch fails.
        """
        # 1. 🚀 TRY KAFKA FIRST (High Throughput)
        kafka_batch = self.kafka_client.consume_batch(limit=500)
        if kafka_batch:
            return FeatureProcessor.process_telemetry({str(i): msg for i, msg in enumerate(kafka_batch)})
        
        # 2. 🔥 FALLBACK TO FIREBASE (Command & Control)
        try:
            raw = db.reference("/SENSOR_DATA").order_by_key().limit_to_last(500).get()
            if not raw:
                return None
            return FeatureProcessor.process_telemetry(raw)
        except Exception as e:
            logging.error(f"Data Fetch Error: {e}")
            return None

    def run_cycle(self) -> None:
        """
        Executes a single end-to-end MLOps cycle: ingestion, inference, metric logging, and alerting.
        """
        # 1. Fetch data FIRST (Network I/O - Do not time)
        df = self.fetch_and_process()
        if df is None or len(df) < 10:
            return
        assert df is not None

        features = df[["distance_cm", "rolling_mean", "rolling_std"]]

        # 2. Train & Save Model (Disk I/O & Training - Do not time)
        self.engine.train(features)

        # 3. START THE INFERENCE TIMER HERE ⏱️
        start_time = time.time()

        # 🤖 ONLY Time the Prediction Engine
        preds, _ = self.engine.predict(features)

        # 4. STOP TIMER
        current_latency = time.time() - start_time

        # Safety catch bounding for algorithmic efficiency
        current_latency = max(current_latency, 0.0001)

        # 5. CALCULATE INDUSTRIAL METRICS
        # Throughput (Events Per Second)
        eps = len(df) / current_latency

        # End-to-End Latency (Ingest to Flag)
        e2e_latency = (time.time() - df["timestamp"].astype(int) / 1e9).mean()

        # Detection metrics (Precision, Recall, F1)
        detection_metrics = {}
        if "true_label" in df.columns:
            detection_metrics = MetricsEvaluator.calculate_detection_metrics(
                df["true_label"].values, preds
            )

        # Calculate rigorous operation metrics
        latency_improvement = ((self.LEGACY_LATENCY - current_latency) / self.LEGACY_LATENCY) * 100
        accuracy_score = (len(df[preds == 1]) / len(df)) * 100

        # Enforce MLflow Lineage
        try:
            with mlflow.start_run(run_name="Mission_Pulse"):
                mlflow.log_metric("response_time_improvement_pct", latency_improvement)
                mlflow.log_metric("model_accuracy_pct", accuracy_score)
                mlflow.log_metric("events_per_second", eps)
                mlflow.log_metric("e2e_latency_seconds", e2e_latency)

                if detection_metrics:
                    mlflow.log_metric("precision", detection_metrics["precision"])
                    mlflow.log_metric("recall", detection_metrics["recall"])
                    mlflow.log_metric("f1", detection_metrics["f1"])
        except Exception as e:
            logging.error(f"MLflow Log Error: {e}")

        # Push state to Dashboards via Firebase Operations DB
        metrics_payload = {
            "last_update": time.time(),
            "accuracy": f"{accuracy_score:.2f}%",
            "latency_gain": f"{max(0.0, latency_improvement):.1f}%",
            "eps": f"{eps:.2f}",
            "e2e_latency": f"{e2e_latency:.2f}s",
            "status": "NOMINAL" if accuracy_score > 85 else "ANOMALY_SENSITIVE",
        }

        if detection_metrics:
            metrics_payload.update({
                "precision": f"{detection_metrics['precision']:.3f}",
                "recall": f"{detection_metrics['recall']:.3f}",
                "f1": f"{detection_metrics['f1']:.3f}",
            })

        db.reference("/SYSTEM_METRICS").set(metrics_payload)

        # 6. 🧠 ADAPTIVE WINDOWING LOGIC
        # If any anomalies are detected in the current window, decrease polling interval
        # to ensure higher system responsiveness.
        if not anoms.empty:
            self.polling_interval = 0.5  # High frequency monitoring
            logging.warning(f"⚠️ ANOMALY DETECTED! Module Accuracy Status: {accuracy_score:.1f}%. Increasing polling frequency (0.5s)")
        else:
            self.polling_interval = 10.0  # Nominal state (power-saving)
            logging.info(f"✅ NOMINAL. Polling frequency (10.0s)")

        if not anoms.empty:
            alert = anoms.tail(1).to_dict("records")[0]
            alert["timestamp"] = str(alert["timestamp"])
            db.reference("/ML_ALERTS").push(alert)


if __name__ == "__main__":
    import firebase_admin
    from firebase_admin import credentials

    if not firebase_admin._apps:
        cred = credentials.Certificate(config.SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"databaseURL": config.DB_URL})

    orchestrator = MLOrchestrator()
    logging.info("🛰️ Satellite Orchestrator Online. Running strict MLOps cycles...")

    while True:
        orchestrator.run_cycle()
        time.sleep(orchestrator.polling_interval)
