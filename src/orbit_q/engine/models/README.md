# Models Module

Anomaly detection models that score health metrics and emit `NOMINAL` / `ANOMALY_SENSITIVE` states.

## Tracked Metrics
- **Power (EPS)**: Voltage, Current Draw, Battery Depth of Discharge.
- **Thermal (TCS)**: Radiator temps, Internal PCB thermistors.
- **Comms**: SNR, Bit Error Rate (BER), Carrier Lock Status.

## Algorithms
We employ **IsolationForest** variants over multivariate time-series windows because they are computationally lightweight and provide explainable decision boundaries, ideal for immediate real-time triage.

## Sample Configuration (`config.toml`)
```toml
[models.isolation_forest]
n_estimators = 100
contamination = 0.05
max_samples = "auto"
random_state = 42

[models.features]
rolling_window_size = 50
normalization = "StandardScaler"
```
