import time
import random
import firebase_admin
from firebase_admin import db, credentials
import config
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not firebase_admin._apps:
    cred = credentials.Certificate(config.SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred, {"databaseURL": config.DB_URL})

faces = ["NORTH", "SOUTH", "EAST", "WEST"]

def transmit():
    logging.info("🚀 Satellite Launch Successful. Transmitting telemetry...")
    while True:
        face = random.choice(faces)
        
        # 5% chance of anomaly
        if random.random() < 0.05:
            distance = random.uniform(300.0, 500.0) 
            logging.warning(f"❗ Simulating Anomaly on {face} face!")
        else:
            distance = random.uniform(20.0, 100.0)

        packet = {
            "face": face,
            "distance_cm": round(distance, 2),
            "timestamp": time.time(),
            "signal_strength": random.randint(70, 100)
        }
        
        db.reference("/SENSOR_DATA").push(packet)
        logging.info(f"Sent: {face} | Dist: {distance:.2f}cm")
        time.sleep(1)

if __name__ == "__main__":
    transmit()