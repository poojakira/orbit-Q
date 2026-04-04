import time
import random
import numpy as np
import firebase_admin
from firebase_admin import db, credentials
from orbit_q import config
import logging
import os
from orbit_q.ingestion.kafka_client import OrbitKafkaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(config.SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"databaseURL": config.DB_URL})
    except Exception as e:
        logging.warning(f"Failed to initialize Firebase (mock mode or missing creds): {e}")


kafka_client = OrbitKafkaClient()
faces = ["NORTH", "SOUTH", "EAST", "WEST"]


def transmit():
    logging.info("🚀 Satellite Launch Successful. Transmitting telemetry...")
    while True:
        face = random.choice(faces)

        # 5% chance of anomaly
        is_anomaly = random.random() < 0.05
        is_missing = random.random() < 0.01
        is_delayed = random.random() < 0.02
        is_corrupted = random.random() < 0.01

        if is_missing:
            logging.warning("⚠️ Simulating missing packet!")
            time.sleep(1)
            continue

        if is_anomaly:
            distance = random.uniform(300.0, 500.0)
            logging.warning(f"❗ Simulating Hardware Anomaly on {face} face!")
        elif is_corrupted:
            distance = float("nan") if random.random() < 0.5 else -9999.0
            logging.warning(f"💥 Simulating Corrupted Data on {face} face!")
        else:
            distance = random.uniform(20.0, 100.0)

        packet = {
            "face": face,
            "distance_cm": distance if np.isnan(distance) else round(distance, 2),
            "timestamp": time.time() - (5.0 if is_delayed else 0.0),  # delay by 5 seconds
            "signal_strength": random.randint(70, 100),
            "true_label": -1 if is_anomaly else 1
        }

        if db.reference:
            try:
                db.reference("/SENSOR_DATA").push(packet)
            except Exception as e:
                logging.debug(f"Firebase push failed (likely mock mode): {e}")

        # --- 🚀 ADDED KAFKA INGESTION ---
        kafka_client.produce_telemetry(packet)

        logging.info(f"Sent: {face} | Dist: {distance:.2f}cm | Label: {'ANOMALY' if is_anomaly else 'NOMINAL'} | Kafka: OK")
        time.sleep(0.5)  # Increased frequency for high-throughput testing


if __name__ == "__main__":
    transmit()
