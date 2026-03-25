# UI / Dashboard Module

Operator frontend written in Streamlit for visualizing real-time telemetry, alert feeds, and historical incident timelines.

## How to Run
```bash
# Launch the primary command center locally
streamlit run dashboard.py
```

Or run the full dev suite via the project root:
```bash
make dev
```

## Available Views
- **Telemetry Stream (Live):** Plots live ingested sensor data from Firebase.
- **KPI Ribbon:** Displays instantaneous state variables (Battery Depth of Discharge, Radiator Temps).
- **Incident Timeline (`pages/`):** A historical ledger of triggered `/ML_ALERTS` and anomaly occurrences.
