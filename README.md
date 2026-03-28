# Orbit-Q — Distributed ML Satellite Telemetry Platform

**Production-grade, GPU-accelerated anomaly detection infrastructure for satellite operations**

[![CI](https://github.com/poojakira/orbit-Q/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/orbit-Q/actions)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)]()
[![Tests](https://img.shields.io/badge/tests-11%20passed-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

---

## 1. Overview

Orbit-Q is a systems-level ML infrastructure platform specifically designed for satellite telemetry anomaly detection. Engineered for production-scale reliability, it provides an end-to-end pipeline from high-frequency telemetry ingestion to multi-model ensemble detection and operator-facing command & control dashboards.

---

## 2. Key Features

- **Performance**: GPU-accelerated ensemble detection with Triton CUDA kernels for nanosecond-level score fusion
- **Resilient Ingestion**: High-throughput REST/gRPC endpoints with automated event-schema mapping and fallback mechanisms
- **Advanced ML**: Multi-model ensemble combining IsolationForest (global outliers), PyTorch Autoencoder (feature manifold), and LSTM (temporal patterns)
- **MLOps Lifecycle**: Automated drift-detection and retraining pipelines with full MLflow lineage tracking
- **Mission Security**: HMAC-SHA256 stream token authentication with comprehensive audit trail logging
- **Command Center**: A 10-page Streamlit suite for live telemetry, mission diagnostics, and performance auditing

---

## 3. Architecture

Orbit-Q follows a decoupled, modular architecture designed for high availability and low latency.

### Package Structure

- `src/orbit_q/`
  - `cli.py`: Main entry point with 6 mission-critical commands
  - `engine/`: Core ML ensemble and custom CUDA kernels for score fusion
  - `ingestion/`: High-frequency telemetry entry point (REST/gRPC)
  - `orchestrator/`: Central rules engine and stream processing coordinator
  - `dashboard/`: Full-stack Streamlit C2 interface
  - `mlflow_tracking/`: Experiment lineage and automated model maintenance
  - `simulator/`: Fault-injection telemetry generators for testing

---

## 4. Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (Required for GPU acceleration features)
- Virtual Environment (Recommended)

### Installation

```bash
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -e .            # Standard installation
pip install -e ".[gpu]"     # Enable GPU acceleration
pip install -e ".[dev]"     # Development tools
```

### Configuration

```bash
ORBIT_Q_SIGNING_SECRET=your-secure-secret-key
MLFLOW_TRACKING_URI=sqlite:///mlruns/orbit_q.db
FIREBASE_DB_URL=https://your-project.firebaseio.com
```

---

## 5. CLI Commands

| Command | Description |
|---|---|
| `orbit-q simulator` | Start a single-satellite mock telemetry stream |
| `orbit-q orchestrator` | Run the ML pipeline and rule-dispatch daemon |
| `orbit-q dashboard` | Launch the Streamlit command center (default :8501) |
| `orbit-q benchmark` | Execute a high-rate throughput and latency stress test |
| `orbit-q stress-test` | Simulate multiple concurrent satellite streams |
| `orbit-q retrain` | Manually trigger the ensemble retraining pipeline |

---

## 6. ML Ensemble

| Model | Type | Role |
|---|---|---|
| **IsolationForest** | Tree ensemble | Global outlier detection |
| **PyTorch Autoencoder** | Neural network | Feature manifold learning |
| **LSTM** | Recurrent network | Temporal pattern modeling |
| **Triton Fusion Kernel** | CUDA kernel | Nanosecond-level score combining |

---

## 7. Design Decisions

| Decision | Rationale |
|---|---|
| `sklearn` → `cuML` fallback | Portability without sacrificing GPU performance on CUDA machines |
| Ensemble: IF + AE + LSTM | IF catches global outliers, AE learns feature manifold, LSTM models temporal context |
| Triton kernel for fusion | Avoids Python overhead for high-frequency (200 Hz+) score combining |
| DDP via `mp.spawn` | SLURM-compatible; no dependency on Horovod/Ray for standard multi-GPU |
| Drift-based retraining | Prevents model staleness under changing sensor calibrations |
| HMAC stream tokens | Stateless auth with TTL; no DB lookup needed for token validation |

---

## 8. Security & Reliability

- **Auth**: Stateless HMAC-SHA256 stream tokens with defined TTL (time-to-live)
- **Graceful Fallback**: Automatic CPU fallback if cuML/GPU components are unavailable
- **Resilient Data**: Logic to handle missing packets, latency jitter, and corrupted (NaN) sensor inputs
- **Audit**: Every detected anomaly and system command is recorded in a tamper-proof audit trail

---

## 9. Operator Dashboard (10-Page Suite)

1. **Live Telemetry**: High-frequency streaming charts for all satellite subsystems
2. **Alert & Command**: Real-time anomaly log with interactive operator intervention tools
3. **Hardware Diagnostics**: Deep-dive into thermal, electrical, and mechanical telemetry
4. **Orbital Tracking**: TLE-based position visualization and signal lock status
5. **Raw Telemetry Logs**: Searchable database of all historical telemetry packets
6. **Performance Audit**: MLOps compliance tracker; accuracy vs. contamination audit
7. **Inference Latency**: Microsecond-level tracking of GPU engine performance
8. **MLflow Lineage**: Full experiment lineage; tracks every mission pulse and model run
9. **Model Retraining**: Manual trigger interface for the ensemble retraining pipeline
10. **Endpoint Health**: Real-time status of the ingestion API and downstream services

---

## 10. Testing

```bash
pytest tests/ -v                         # Run core test suites
pytest tests/ --cov=src --cov-report=html  # Generate coverage report
```

### Verified Test Suites

- **ML Engine**: Ensemble initialization, cross-validation, and prediction accuracy
- **Simulator**: Packet schema integrity and fault-injection accuracy
- **Security**: HMAC validation, token expiry, and unauthorized access prevention

---

## 11. License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 12. Team Contributions

### Pooja Kiran — Lead ML Systems Engineer & Core Architect

| # | Contribution Area | Details | Quantified Impact |
|---|---|---|---|
| 1 | Multi-Model ML Ensemble Engine | Designed and implemented 3-model ensemble: IsolationForest (global outliers), PyTorch Autoencoder (feature manifold), LSTM (temporal patterns) | 3 models fused; ensemble handles 200 Hz+ satellite telemetry streams |
| 2 | Custom Triton CUDA Fusion Kernel | Engineered CUDA kernel (`engine/`) for nanosecond-level score fusion across all 3 ensemble models | Nanosecond-level fusion latency; avoids Python GIL overhead at high frequency |
| 3 | cuML / CPU Graceful Fallback | Implemented automatic fallback from cuML GPU to sklearn CPU when CUDA unavailable | 100% portability across GPU and CPU environments |
| 4 | DDP Multi-GPU Training | Implemented PyTorch Distributed Data Parallel (DDP) via `mp.spawn` for SLURM-compatible multi-GPU training | SLURM-compatible; no Horovod/Ray dependency |
| 5 | MLflow Experiment Lineage | Built full MLflow tracking system for every model training run, drift event, and retraining trigger | Full lineage tracking; automated retraining at 0.1 KL-divergence drift threshold |
| 6 | Drift Detection & Auto-Retraining | Implemented statistical drift detection pipeline with KL-divergence monitoring and auto-retrain triggers | Prevents model staleness under changing sensor calibrations |
| 7 | HMAC-SHA256 Stream Authentication | Designed stateless HMAC stream token authentication with defined TTL and comprehensive audit logging | Stateless auth; every anomaly and command recorded in tamper-proof audit trail |

### Rhutvik Pachghare — Distributed Systems & Mission Operations Engineer

| # | Contribution Area | Details | Quantified Impact |
|---|---|---|---|
| 1 | Mission Simulation Engine | Built fault-injection telemetry simulator (`simulator/`) generating realistic satellite sensor streams with configurable anomaly injection | Supports multiple concurrent satellite streams via `orbit-q stress-test` |
| 2 | Distributed Orchestrator | Engineered the central rules engine and stream processing coordinator (`orchestrator/`) managing the ML pipeline daemon | Real-time dispatch; handles missing packets, latency jitter, and NaN sensor inputs |
| 3 | 10-Page Streamlit C2 Dashboard | Developed the full-stack Streamlit Command & Control interface across 10 specialized mission control modules | 10 pages: live telemetry, alert/command, diagnostics, orbital tracking, logs, audit, latency, MLflow, retraining, endpoint health |
| 4 | REST/gRPC Ingestion Layer | Implemented high-throughput telemetry ingestion endpoints (`ingestion/`) with automated event-schema mapping | Supports 200 Hz+ ingestion rate with gRPC buffering + REST load balancing fallback |
| 5 | Black Formatting & Code Quality | Applied Black formatting and standardized code style across the codebase; committed as `Apply black formatting from friend` | Uniform code style across all modules |

---

**License**: MIT | **Platform**: Python 3.9+ | **GPU**: CUDA 11.8+
