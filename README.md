# 🛰️ CubeSat 3D Anomaly Engine

**OrbitIQ Mission Control: Autonomous Satellite Health &amp; Telemetry MLOps.**

The CubeSat 3D Anomaly Engine is a specialized MLOps ecosystem built for high-frequency orbital monitoring. By integrating Isolation Forest unsupervised learning with a Firebase ground station, the platform detects spatial anomalies—such as proximity to space debris or sensor degradation—with sub-second latency. The suite features a 10-page command center for full operational visibility over satellite hardware and AI performance.

📊 Technical Performance Report

Benchmarks captured during the current mission cycle (referenced from ml_orchestrator.py and 07_Inference_Latency.py):

Latency Optimization:

Legacy Baseline: 2000.0ms (Standard cloud-polling)

Engine Inference: ~8.0ms (Edge-optimized Isolation Forest)

Net Latency Reduction: 99.9% (Achieved via rolling window feature extraction)

Detection Integrity:

Predictive Accuracy: 95.8% (Target: >95.0%)

Anomaly Sensitivity: 0.05 Contamination Rate (Configured for high-precision outlier detection)


🚀 Key Features
1. Mission Control Dashboard (dashboard.py)
The primary strategic interface for satellite operations.

📈 Live Telemetry Stream: Real-time polling and visualization of sensor distances across all satellite faces (N, S, E, W).

🚨 Alert Command: AI-driven triage center highlighting high-risk anomalies for immediate operator intervention.

📡 Orbital Radar: A 360° polar visualization of spatial proximity to detect incoming objects or debris.

2. AI & MLOps Infrastructure
The autonomous backbone responsible for data transformation and governance.

🧠 Isolation Forest Engine: Specialized outlier detection trained on rolling mean and standard deviation features to identify non-linear sensor patterns.

📜 MLflow Lineage: Full experiment tracking and audit logs, ensuring every model version is benchmarked for accuracy and speed.

🔄 Mission Orchestrator: A continuous "Mission Pulse" coordinating the fetch-train-predict cycle every 10 seconds.

3. Maintenance & Health Suite
⚙️ Hardware Diagnostics: Real-time monitoring of internal system health (CPU load, temperature, and power stability).

🩺 Endpoint Health: Infrastructure monitoring for container RAM usage, Firebase connectivity, and model weight availability.

🔄 Manual Retraining: Operator-triggered model updates using the latest telemetry batches to adapt to new orbital environments.



🛠️ Installation & Setup

NOTE: in firebase download your service account file to start 
Prerequisites

Python 3.10+

Firebase Service Account Key (service_account.json)

SQLite (for local MLflow tracking)

Step-by-Step

Note:Extract .zip file then follow this steps

1. Install Dependencies

                         pip install -r requirements.txt


2. Initialize Sidebar Pages (Required for Firebase context)

                                                             python fix_pages.py

3. ## 🚀 Quick Start & Developer Setup

Because this project uses a live database to stream telemetry, you will need to connect it to your own free Firebase instance to run it locally. 

### 1. Set up Firebase (The Backend)
1. Go to the [Firebase Console](https://console.firebase.google.com/) and create a new free project.
2. Navigate to **Build > Realtime Database** and click **Create Database**.
3. Go to **Project Settings** (the gear icon) > **Service Accounts**.
4. Click **Generate New Private Key**. 
5. Save this downloaded file in the root directory of this project and rename it to exactly `service_account.json`.

### 2. Update Configuration
Open `config.py` in your code editor and replace the `FIREBASE_DB_URL` with your new database URL (found at the top of your Firebase Realtime Database page):


# config.py
DB_URL = os.getenv("FIREBASE_DB_URL", "[https://YOUR-NEW-PROJECT-ID.firebasedatabase.app/](https://YOUR-NEW-PROJECT-ID.firebasedatabase.app/)")



   🚦 How to Run

<img width="372" height="117" alt="image" src="https://github.com/user-attachments/assets/7ae6e87b-2f72-45ce-882f-84b01b1ab00f" />

   📂 Project Architecture

<img width="1021" height="626" alt="image" src="https://github.com/user-attachments/assets/bc6e47c1-c9c8-4038-ab5c-a11643897b44" />


    ⚖️ License & Classification
SATELLITE OPERATIONS // PROPRIETARY // INTERNAL USE ONLY © 2026 CubeSat 3D Anomaly Engine. All rights reserved.


![dashboard_orbit_command](https://github.com/user-attachments/assets/ca586a8d-13a9-4a04-8d69-686f829b3516)

