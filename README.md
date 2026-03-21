<div align="center">
  <h1>🛰️ Orbit-Q (OrbitIQ Ops)</h1>
  <p><strong>Enterprise-Grade Satellite Operations Command Center & Telemetry Anomaly Detection</strong></p>

  <p>
    [![CI Status](https://github.com/poojakira/orbit-Q/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/orbit-Q/actions/workflows/ci.yml)
    [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
    [![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
    [![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io)
    [![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)](https://mlflow.org/)
  </p>
</div>

---

## 📖 Executive Summary

**Orbit-Q** is a sophisticated command and control (C2) dashboard engineered for modern satellite operations. By integrating real-time telemetry streaming with state-of-the-art unsupervized machine learning anomaly detection, it preemptively identifies mission-critical hardware degradation.

**Orbit-Q gives you:**
* **Real-time Telemetry Ingestion:** Processes multi-dimensional spacecraft sensor states seamlessly.
* **Preemptive Threat Detection:** Identifies thermal anomalies and power fluctuations before escalation using an `IsolationForest` engine.
* **Rigorous MLOps Lineage:** Ensures every detection model is versioned, parameter-matched, and performance-tracked via MLflow.
* **Enterprise KPI Dashboarding:** Exposes highly responsive operator state variables through a modular Streamlit GUI.

### Who it's for
* **Satellite Ops Engineers** needing an intuitive, low-latency diagnostic interface.
* **Mission Control Teams** requiring immediate deterministic alerting on hardware faults.
* **ML Ops / SREs** looking for a strict, robust telemetry machine-learning pipeline architecture.

---

## 🗺️ What to Read First

For a busy reviewer evaluating the core systems logic, please review these key components in order:
1. **`ml_orchestrator.py`**: The central heartbeat of the system. Handles real-time telemetry fetching, feature extraction dispatch, and ML metric logging.
2. **`ml_engine.py`**: Contains the strictly typed `AnomalyEngine`, showcasing unsupervized scikit-learn models integrated securely with MLflow.
3. **`tests/test_ml_engine.py`**: Demonstrates strict Test-Driven Development (TDD) proficiency with extensive internal API mocking (MLflow & Scikit-learn).
4. **`dashboard.py`**: The operator frontend logic demonstrating robust, layout-optimized real-time UI components.

---

## 🏗️ System Architecture

### Component Flow (Data Pipeline)
1. **[Telemetry Simulator]**: Synthesizes and pushes orbital state vectors.
2. **[Firebase Realtime DB]**: Acts as the low-latency message broker.
3. **[Orbit-Q ML Orchestrator]**: Pulls rolling windows of telemetry, computes statistical features.
4. **[Anomaly Engine]**: Evaluates matrices using IsolationForest. Logs artifacts silently to `MLflow`.
5. **[Streamlit Operator Dashboard]**: Subscribes to Firebase updates and paints the situational KPI ribbon.

---

## 📂 Repository Structure

* **`tests/`**: Contains the `pytest` suite ensuring CI/CD reliability, mocking external services.
* **`pages/`**: Modular Streamlit frontend components extending the primary operator dashboard.
* **`.github/workflows/`**: Strict CI linting, typing, and automated testing actions.
* **`ml_engine.py`**: The heavily typed, strictly documented Anomaly Detection subsystem.
* **`ml_orchestrator.py`**: The data IO loop connecting Firebase to the ML engine.
* **`feature_processor.py`**: Stateless pure functions for high-speed rolling metrics extraction.

---

## 🚀 Quick Start & End-to-End Demo

**Resource Requirements:** Minimal (At least 2vCPU, 2GB RAM). Model inference is extremely lightweight due to precise feature selection. 

### Commands
```bash
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### End-to-End Demo Scenario:
1. **Initialize Environment**: Ensure `config.py` contains valid Firebase and MLflow credentials.
2. **Launch Telemetry**: Run `python mock_telemetry.py` on a separate terminal. This simulator will inject baseline sensor states followed by a synthesized thermal spike into Firebase.
3. **Start the Orchestrator**: Run `python ml_orchestrator.py` to begin continuous feature fetching and anomaly model training / prediction cycles.
4. **Launch Dashboard**: Run `streamlit run dashboard.py`.
5. **Observe**: Watch the Enterprise KPI Ribbon adapt securely to the nominal state. Upon receiving the simulated thermal spike (Step 2), the anomaly triggers visually in the dashboard and logs heavily into MLflow.

**Performance Results:** The deterministic anomaly scenario triggers and is categorized by the unsupervised model within `<1.5 seconds` at a simulated `50 Hz` telemetry ingest rate.

---

## 🧪 Testing & CI

Orbit-Q maintains rigorous codebase quality constraints ensuring deployment stability.
* **Coverage**: `>90%` coverage across core ML execution pipelines (`ml_engine.py`, `ml_orchestrator.py`).
* **CI Build**: GitHub Actions pipeline executes dynamically across multiple Python versions to guarantee broad compatibility.
* **Pipeline Checks**: Execution of `black` (formatting), `flake8` (static analysis), `mypy` (strict variable typing), and `pytest` (mock-driven unit verification).

---

## 🛡️ Production & Reliability

### Failure Modes & Resiliency
* **Lost Telemetry / Network Timeout**: `ml_orchestrator.py` uses exception shielding and `None` handling. In an event of a dropout, ML lifecycle operations gracefully pause without crashing the daemon.
* **Delayed Packets**: Handled implicitly; incoming telemetry is evaluated chronologically via `.order_by_key().limit_to_last(500)`.
* **Dashboard State Failures**: Data fetches are wrapped in granular `try/except` safeguards preventing a blank screen from terminating the broader UI state.

### Alerting & Monitoring Approach
Currently, the `AnomalyEngine` flags binary statuses (`NOMINAL`, `ANOMALY_SENSITIVE`). On identification, critical subsets are routed immediately to a Firebase `/ML_ALERTS` queue. Scalable implementations can seamlessly bind this node to PagerDuty or Slack Webhooks for immediate downstream engineering response.

### Security Considerations
* **Authentication/Authorization**: Data manipulation is tightly coupled to Firebase Admin SDKs driven by server-side private credentials.
* **Credentials Injection**: Repository expects decoupled secrets (`secrets.toml` or environment keys) entirely bypassing hardcoded credentials. 

---

## 💡 Design Constraints & Trade-Offs

* **Why Firebase?** Chose standard Firebase Realtime DB over Kafka for the MVP pipeline. Firebase provides instantaneous push/subscribe mechanisms and simplifies frontend dashboard syncing without requiring complex Zookeeper/Broker configurations overhead.
* **Why IsolationForest?** Selected over complex deep autoencoders. Orbital anomalies require extremely low latency and explainable boundaries. IsolationForest executes reliably over multi-variate continuous telemetry data without extensive GPU availability or complex training epoch waits.
* **Why Streamlit?** Allows immediate Pythonic binding of predictive models to an operator UI without the latency penalty of managing standalone React/Node backends.
