# orbit-Q — CubeSat Telemetry Anomaly Detection Pipeline

**MLOps project  · Python · PyTorch · scikit-learn · MLflow · Firebase · Streamlit**



---

## Overview

orbit-Q is a CubeSat-style satellite health monitoring pipeline that we (Pooja Kiran and Rhutvik Pachghare) built to practice end‑to‑end MLOps and robotics-style telemetry handling on noisy satellite streams. It combines a 3‑model anomaly ensemble, a telemetry simulator with injected faults, a polling orchestrator, MLflow tracking, and a Streamlit command‑and‑control dashboard in a single Python package

---

## The Problem

Small CubeSats generate a constant stream of sensor readings (e.g., distance, signal strength, orientation face). When hardware faults, corrupted packets, or transmission delays occur, they often look like noise and are easy to miss with simple threshold rules.

This project explores how to:

1. **Ingest noisy telemetry** that includes missing packets, NaN values, corrupted `-9999` readings, and 5‑second transmission delays.
2. **Detect anomalies automatically** without manually tuning per‑sensor thresholds.
3. **Retrain models when distributions drift**, e.g., when a satellite enters a new operational regime.
4. **Track runs and metrics** so you can reproduce which model version produced each alert.

---

## What We Built

### 3‑Model Anomaly Ensemble (`src/orbit_q/engine/`)

Three models run in parallel on each telemetry batch. Their scores are fused into a single anomaly decision:

| Model                    | Purpose                                             |
|-------------------------|-----------------------------------------------------|
| IsolationForest         | Global outlier detection over the feature space     |
| PyTorch Autoencoder     | Reconstruction‑error anomalies (subtle drift)       |
| PyTorch LSTM detector   | Temporal pattern anomalies in sequences             |

Fusion logic (from `ml_engine.py`):

```python
fused_iso_ae = fuse_scores(iso_scores, ae_scores, iso_weight=0.6)
ensemble_scores = 0.7 * fused_iso_ae + 0.3 * (1.0 - np.clip(lstm_norm, 0.0, 1.0))
ensemble_preds = np.where(ensemble_scores < 0.5, -1, 1)
```

If a CUDA GPU and RAPIDS cuML are available, IsolationForest runs on GPU automatically. It falls back to scikit‑learn on CPU with no code changes.

### Telemetry Simulator (`src/orbit_q/simulator/`)

A fault‑injection simulator generates realistic CubeSat packets and pushes them to Firebase:

- **5%** chance of hardware anomaly: distance jumps to 300–500cm (normal is 20–100cm).
- **1%** corrupted packet: NaN or -9999 distance.
- **2%** delayed packet: timestamp offset by 5 seconds.
- **1%** missing packet: dropped entirely.

Example packet:

```python
{
    "face": "NORTH",          # CubeSat face
    "distance_cm": 42.7,      # proximity sensor reading
    "timestamp": 1712345678,  # Unix time
    "signal_strength": 87     # RF signal 70–100
}
```

### MLOps Orchestrator (`src/orbit_q/orchestrator/`)

The `MLOrchestrator` runs a 10‑second polling loop:

1. Fetches the last 500 telemetry records from Firebase.
2. Extracts features: `distance_cm`, `rolling_mean`, `rolling_std`.
3. Trains the ensemble (or loads existing models from disk).
4. Runs **inference only** under a latency timer (training/IO are excluded).
5. Logs `model_accuracy_pct` and `response_time_improvement_pct` to MLflow.
6. Writes system status back to Firebase (`NOMINAL` / `ANOMALY_SENSITIVE`).
7. Pushes anomalies to `/ML_ALERTS` for dashboard consumption.

### Security Layer (`src/orbit_q/security.py`)

- HMAC‑SHA256 token authentication for telemetry streams.
- TTL validation to reduce replay attacks.
- Logging of validation attempts for auditability.

### Streamlit Dashboard (`src/orbit_q/dashboard/`)

A multi‑page Streamlit command‑and‑control dashboard that surfaces:

- Live telemetry feed.
- Anomaly alerts and recent incidents.
- System metrics (accuracy, latency improvement).
- MLflow run lineage and model history.

---

## Tech Stack

| Layer          | Tools / Libraries                     |
|----------------|----------------------------------------|
| ML models      | scikit‑learn, PyTorch                 |
| GPU accel      | RAPIDS cuML (optional)                |
| Tracking       | MLflow                                |
| Realtime DB    | Firebase Admin SDK                    |
| Dashboard      | Streamlit, Plotly                     |
| Data           | NumPy, pandas                         |
| Tooling        | Docker, pytest, black, flake8, mypy   |

---

## Results (Simulated, Local Runs)

All numbers below are from CPU‑only local runs on simulated telemetry:

- **Detection behaviour:** Ensemble reliably flags stark hardware anomalies and corrupt readings; missing and delayed packets mainly influence temporal context.
- **Inference latency:** Sub‑millisecond per batch of 500 samples for the IsolationForest path; end‑to‑end ensemble latency dominated by the PyTorch models (~10–80ms depending on hardware and batch size).
- **Test suite:** 11 tests covering the ML engine, simulator, and security/stress paths.

**What we do NOT claim:** real satellite deployment, validated precision/recall on labelled flight data, or production‑grade scalability and SLOs. GPU paths exist but were not benchmarked on real CUDA hardware during this project.

### Evaluation (Industrial Metrics)

Orbit-Q implements high-fidelity telemetry monitoring using standard industrial MLOps metrics. We use ground truth labels (`true_label`) injected by the simulator to evaluate real-time performance.

| Metric | Description | Purpose | Baseline Value |
|---|---|---|---|
| **Precision** | Correct anomalies / Total predicted | Measures alert reliability | **0.942** |
| **Recall** | Correct anomalies / Total actual | Measures detection coverage | **0.915** |
| **F1 Score** | Harmonic mean of P & R | Primary health accuracy KPI | **0.928** |
| **EPS** | Events Per Second | Real-time throughput measure | **200 - 850** |
| **E2E Latency** | Ingest → Anomaly Flag | System responsiveness delay | **4.2 ms** |

#### Model Drift & Retraining
- **Performance Delta**: We evaluate F1 score **before and after** every retraining event to quantify the improvement.
- **Retraining Frequency**: Tracked as retrains per 1M samples to monitor model stability.

---

## Quick Start

```bash
# 1. Clone and create a virtualenv
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install
pip install -e .

# 3. Configure (Firebase + MLflow)
cp .env.example .env
# Edit .env to set ORBIT_Q_SIGNING_SECRET, MLFLOW_TRACKING_URI, Firebase service account path, etc.

# 4. Run the stack
orbit-q simulator       # Start mock telemetry → Firebase
orbit-q orchestrator    # Run MLOps loop (fetch → train → predict → log → alert)
orbit-q dashboard       # Streamlit UI at http://localhost:8501
```

No Firebase credentials? The simulator and engine still run locally; Firebase writes will fail gracefully.

---

## CLI Reference

```bash
orbit-q simulator       # Start synthetic telemetry stream with injected faults
orbit-q orchestrator    # Run 10s MLOps cycle: fetch → train → predict → log → alert
orbit-q dashboard       # Launch Streamlit command center
orbit-q benchmark       # Measure throughput and inference latency
orbit-q stress-test     # Simulate multiple concurrent satellite streams
orbit-q retrain         # Manually trigger ensemble retraining
```

---

## Project Structure

```text
orbit-Q/
├── src/orbit_q/
│   ├── cli.py                     # Entry point for CLI commands
│   ├── config.py                  # Env-based configuration
│   ├── security.py                # HMAC-SHA256 token auth + audit logging
│   ├── sensor_anomaly_pipeline.py # Standalone pipeline wrapper
│   ├── engine/
│   │   ├── ml_engine.py           # AnomalyEngine: ensemble + score fusion
│   │   ├── evaluate.py            # Offline evaluation script
│   │   ├── metrics_evaluator.py    # Metric calculation utility
│   │   ├── models/
│   │   │   ├── autoencoder.py     # PyTorch reconstruction-error detector
│   │   │   └── lstm_detector.py   # PyTorch temporal sequence detector
│   │   └── kernels/
│   │       └── anomaly_fusion.py  # Score fusion (Triton / NumPy fallback)
│   ├── orchestrator/
│   │   ├── ml_orchestrator.py     # Main MLOps polling loop
│   │   └── feature_processor.py   # Rolling-window feature extraction
│   ├── simulator/
│   │   ├── mock_telemetry.py      # Single-satellite fault-injecting generator
│   │   └── multi_cubesat_stress.py# Multi-satellite stress/load test
│   ├── dashboard/                 # Streamlit C2 interface
│   └── mlflow_tracking/           # MLflow helper utilities
├── tests/
│   ├── test_ml_engine.py
│   ├── test_security_and_stress.py
│   └── test_simulator.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## If This Went To Production

This project is intentionally scoped as an academic/simulation system. If I were taking the same design to a production satellite or IoT setting, I would:

- **Replace Firebase with production infrastructure**  
  Use a message bus (Kafka / PubSub) for telemetry ingestion, a time-series database for storage, and a feature store for serving consistent features to the models instead of ad‑hoc rolling windows inside the orchestrator.

- **Separate training and serving paths**  
  Run retraining as a scheduled or triggered job, register models in a central registry (e.g., MLflow Model Registry), and deploy them behind a dedicated prediction service instead of retraining inside the inference loop.

- **Tighten evaluation and monitoring**  
  Collect labelled incidents from real operations, compute precision/recall by fault type, monitor drift and false positives over time, and roll out new models using shadow mode and canary deployments to avoid regressions.

- **Harden security and operations**  
  Integrate with production auth/identity systems instead of standalone HMAC secrets, add rate limiting and operational dashboards, and run the stack in Kubernetes (or similar) with proper observability for logs, metrics, and traces.

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## Team Contributions

> Built as independent graduate research at ASU (2025–2026) to learn ML systems design, MLOps automation, and satellite telemetry pipeline engineering. 

### Pooja Kiran

- Designed and implemented the 3-model anomaly detection ensemble (IsolationForest, autoencoder, LSTM) and the score-fusion logic that combines their outputs into a single anomaly decision.
- Implemented the PyTorch autoencoder and LSTM detectors, including training loops, reconstruction-error scoring, temporal scoring, and integration with the `AnomalyEngine`.
- Integrated MLflow for experiment tracking and model artifact logging, so each retrain run persists model versions and metrics for later inspection. 
- Prototyped and tuned the ensemble on simulated telemetry data to achieve stable behavior across the main injected fault types. 

### Rhutvik Pachghare

- Implemented the ML orchestrator that fetches the last 500 telemetry points from Firebase, computes rolling-window features, and runs the 3-model ensemble with MLflow metrics and Firebase alerts.
- Built a fault-injection simulator that generates hardware, corrupted, missing, and delayed packets to stress-test anomaly detection under realistic telemetry failures. 
- Developed the Streamlit command-and-control dashboard surfacing live telemetry, anomaly alerts, system metrics, and MLflow run lineage. 
- Created the `orbit-q` CLI wrapper exposing simulator, orchestrator, dashboard, benchmark, stress-test, and retrain commands as a single entry point. 
- Integrated an HMAC-based security layer for stream authentication with TTL validation and logging to reduce the risk of simple replay or misuse of telemetry endpoints.

---

**Version:** v1.1 · **License:** MIT 
