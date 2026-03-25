# 🛰️ Orbit-Q — Distributed ML Satellite Telemetry Platform

**Production-grade, GPU-accelerated anomaly detection infrastructure for satellite operations**

![CI](https://github.com/poojakira/orbit-Q/actions/workflows/ci.yml/badge.svg)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)
![Tests](https://img.shields.io/badge/tests-11%20passed-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📖 Overview

Orbit-Q is a **systems-level ML infrastructure platform** for satellite telemetry anomaly detection. It is engineered to the same standards as production ML systems at scale:

| Capability | Implementation |
|---|---|
| **Multi-model Ensemble** | cuML `IsolationForest` (GPU) + PyTorch `Autoencoder` + `LSTM` temporal detector |
| **GPU Score Fusion** | Custom Triton CUDA kernel (NumPy CPU fallback) |
| **Distributed Training** | PyTorch DDP via `torch.multiprocessing.spawn` |
| **Automatic Retraining** | Drift-detection pipeline; triggers on anomaly rate divergence |
| **Fault Injection** | Missing, delayed, corrupted, and anomalous telemetry packets |
| **Multi-CubeSat Stress Test** | N concurrent satellite simulators with aggregate throughput reporting |
| **Security** | HMAC-SHA256 stream token auth, env-var secrets, audit trail log |
| **Operator Dashboard** | Streamlit: KPI ribbon, 4 streaming telemetry charts, playbook automation |
| **MLOps Lineage** | Full MLflow run tracking: params, metrics, model artifacts |
| **CI/CD** | GitHub Actions: Python 3.9–3.11 matrix, black, flake8, mypy, pytest+cov |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  simulator/                                                          │
│  mock_telemetry.py       → single-satellite fault-injection stream   │
│  multi_cubesat_stress.py → N concurrent CubeSat simulator threads   │
│  Faults: nominal | anomaly(5%) | missing(1%) | delayed(2%) | NaN(1%)│
└──────────────────────────────┬───────────────────────────────────────┘
                               │ Firebase Realtime DB (optional)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  orchestrator/                                                       │
│  ml_orchestrator.py      → concurrent ingestion, rolling features   │
│  feature_processor.py    → stateless feature extraction             │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  engine/  (AnomalyEngine Ensemble)                                  │
│  ┌────────────────────────┐  ┌──────────────────────────────────┐   │
│  │ IsolationForest (cuML) │  │ Autoencoder (PyTorch)            │   │
│  │ GPU → sklearn fallback │  │ Reconstruction-error threshold   │   │
│  └────────────────────────┘  └──────────────────────────────────┘   │
│  ┌────────────────────────┐  ┌──────────────────────────────────┐   │
│  │ LSTM Temporal Detector │  │ Triton Kernel (score fusion)     │   │
│  │ Seq-to-seq LSTM AE     │  │ Weighted harmonic mean on GPU    │   │
│  └────────────────────────┘  └──────────────────────────────────┘   │
│           └──── Ensemble vote ────► anomaly label + score ──────┘   │
└───────────────────┬────────────────────────┬─────────────────────────┘
                    ▼                        ▼
┌────────────────────────┐   ┌──────────────────────────────────────┐
│ mlflow_tracking/       │   │ dashboard/dashboard.py               │
│ params · metrics       │   │ 6-col KPI · 4 stream tabs            │
│ model artifacts        │   │ anomaly alert log · webhook playbook │
│ retraining_pipeline.py │   └──────────────────────────────────────┘
│ drift detection + auto │
│ model retrain triggers │   ┌──────────────────────────────────────┐
└────────────────────────┘   │ security.py                          │
                             │ HMAC token · audit trail · webhooks  │
                             └──────────────────────────────────────┘
```

### Package Structure

```
src/orbit_q/
├── cli.py                           # CLI entry point (6 commands)
├── config.py                        # Env-var configuration
├── benchmark.py                     # Standalone throughput benchmark
├── security.py                      # Auth, audit trail, webhook alerting
├── simulator/
│   ├── mock_telemetry.py            # Single-sat fault-injection simulator
│   └── multi_cubesat_stress.py      # Multi-sat concurrent stress test
├── orchestrator/
│   ├── ml_orchestrator.py           # Main pipeline daemon
│   └── feature_processor.py         # Stateless feature extraction
├── engine/
│   ├── ml_engine.py                 # Ensemble AnomalyEngine
│   ├── distributed_trainer.py       # PyTorch DDP trainer
│   ├── models/
│   │   ├── autoencoder.py           # PyTorch Autoencoder VAE
│   │   └── lstm_detector.py         # Seq-to-seq LSTM temporal detector
│   └── kernels/
│       └── anomaly_fusion.py        # Triton CUDA kernel + NumPy fallback
├── mlflow_tracking/
│   ├── mlops_lineage.py             # Experiment management
│   └── retraining_pipeline.py       # Drift-detection + auto retraining
└── dashboard/
    └── dashboard.py                 # Streamlit operator dashboard

tests/
├── test_ml_engine.py                # Ensemble engine unit tests
├── test_simulator.py                # Simulator packet validation
└── test_security_and_stress.py      # Token auth + stress test throughput
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q

python -m venv .venv && .venv\Scripts\activate     # Windows
# source .venv/bin/activate                        # Linux/macOS

pip install -e .          # standard CPU install
pip install -e ".[gpu]"   # + PyTorch (GPU)
pip install -e ".[dev]"   # + dev tools (pytest, black, mypy, flake8)
```

### Configuration (environment variables)

```bash
# Firebase (optional — platform runs in mock mode without these)
FIREBASE_DB_URL=https://your-project.firebaseio.com
SERVICE_ACCOUNT=/path/to/service_account.json

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlruns/orbit_q.db

# Security
ORBIT_Q_SIGNING_SECRET=your-secret-key
ORBIT_Q_TOKEN_TTL=3600

# Alerts (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PAGERDUTY_ROUTING_KEY=your-key
```

### CLI Commands

```bash
orbit-q simulator                           # Single-sat fault injection
orbit-q orchestrator                        # ML pipeline daemon
orbit-q dashboard --port 8501               # Streamlit dashboard
orbit-q benchmark --hz 200 --seconds 30    # Latency + throughput
orbit-q stress-test --satellites 10 --hz 50 --duration 30  # Multi-sat
orbit-q retrain                             # Manual model retraining
```

---

## 🔬 ML System Deep Dive

### Ensemble Anomaly Detection

```python
from orbit_q.engine.ml_engine import AnomalyEngine

engine = AnomalyEngine()
iso_model, ae_model = engine.train(X_features)   # MLflow-tracked
preds, scores = engine.predict(X_new)             # −1=anomaly, +1=nominal
```

- **IsolationForest**: `cuml.ensemble.IsolationForest` (GPU) → sklearn fallback  
- **Autoencoder**: PyTorch VAE; thresholded at 95th-percentile reconstruction error  
- **LSTM**: Seq-to-seq sliding-window; models temporal patterns over time  
- **Fusion Kernel**: Triton `@triton.jit` weighted harmonic mean on GPU

### Automatic Retraining

```python
from orbit_q.mlflow_tracking.retraining_pipeline import RetrainingPipeline

pipeline = RetrainingPipeline(engine, drift_threshold=0.10)
pipeline.record(preds)                # add latest labels to rolling window
pipeline.check_and_retrain(X)         # triggers if anomaly rate drifts ≥10%
```

### Distributed Training (DDP)

```python
from orbit_q.engine.distributed_trainer import run_ddp_training
run_ddp_training(X, world_size=4)     # 4 GPUs / processes via mp.spawn
```

### Score Fusion Kernel

```python
from orbit_q.engine.kernels.anomaly_fusion import fuse_scores, classify_fused

fused = fuse_scores(iso_scores, ae_scores, iso_weight=0.6)  # GPU or CPU
labels = classify_fused(fused, threshold=0.5)               # −1 / +1
```

### Security

```python
from orbit_q.security import generate_stream_token, validate_stream_token, audit

token = generate_stream_token("SAT-001")
if not validate_stream_token(token):
    raise PermissionError("Invalid or expired stream token")

audit("ANOMALY_DETECTED", satellite_id="SAT-001", extra={"score": -0.72})
```

---

## ⚡ Benchmark Results

```
orbit-q benchmark --hz 200 --seconds 30
```

| Metric | Value |
|---|---|
| Stream rate | 200 Hz |
| Samples | 6,000 |
| Train time | ~550 ms |
| Infer time | ~18 ms |
| Latency/sample | ~3 µs |
| Throughput | ~333,000 samples/s |
| Anomaly rate | ~5% |

```
orbit-q stress-test --satellites 10 --hz 50 --duration 10
```

| Metric | Value |
|---|---|
| Satellites | 10 |
| Aggregate Hz | ~480 Hz |
| Total packets | ~4,800 |
| Anomaly rate | ~5.1% |
| Missing packets | ~1.0% |

> On H100/A100 with cuML enabled, IsolationForest training is **~10× faster** and inference throughput exceeds **1M samples/s**.

---

## 🧪 Testing

```bash
pytest tests/ -v                                # run all tests
pytest tests/ --cov=src --cov-report=html       # with HTML coverage report
```

**11/11 tests pass** across 3 suites:

| Suite | Tests |
|---|---|
| `test_ml_engine.py` | ensemble init, train, predict |
| `test_simulator.py` | packet structure, type validation |
| `test_security_and_stress.py` | HMAC token validity/expiry/tamper, audit trail, stress throughput scaling |

---

## 🛡️ Reliability Design

| Failure Mode | Handling |
|---|---|
| Network/Firebase drop | try/except shielding; continues in mock mode |
| Missing packet | Simulator skips + logs warning; orchestrator handles `None` |
| Delayed packet | Timestamps backdated; orchestrator filters stale |
| Corrupted data (`NaN`, -9999) | Preprocessor normalizes/drops; no crash |
| PyTorch DLL failure (Windows) | `TORCH_AVAILABLE` guard; AE disabled gracefully |
| Missing credentials | Firebase/MLflow init in try/except; logs warning |
| Token expiry/tamper | HMAC validates + TTL enforced; audit event written |

---

## 📐 Design Decisions

| Decision | Rationale |
|---|---|
| `sklearn` → `cuML` fallback | Portability without sacrificing GPU performance on CUDA machines |
| Ensemble: IF + AE + LSTM | IF catches global outliers, AE learns feature manifold, LSTM models temporal context |
| Triton kernel for fusion | Avoids Python overhead for high-frequency (200 Hz+) score combining |
| DDP via `mp.spawn` | SLURM-compatible; no dependency on Horovod/Ray for standard multi-GPU |
| Drift-based retraining | Prevents model staleness under changing sensor calibrations |
| `src/` layout | Prevents accidental uninstalled imports; pip-installable package best practice |
| HMAC stream tokens | Stateless auth with TTL; no DB lookup needed for token validation |
