# --- AUTO-ADDED FIREBASE INIT ---
import firebase_admin
from firebase_admin import credentials, db
from orbit_q import config

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(config.SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"databaseURL": config.DB_URL})
except Exception:
    pass
# --------------------------------

import streamlit as st
import time
from orbit_q.orchestrator.ml_orchestrator import MLOrchestrator

st.set_page_config(page_title="Model Retraining", layout="wide")
st.title("🔄 Pipeline & Retraining Triggers")
st.markdown("Manually trigger an Isolation Forest retrain using the latest telemetry batch.")

orch = MLOrchestrator()

st.subheader("Current Model Configuration")
col1, col2, col3 = st.columns(3)
col1.metric("Algorithm", "Isolation Forest")
col2.metric("Contamination Rate", "0.05")
col3.metric("N_Estimators", "200")

st.divider()

if st.button("🚀 Trigger Manual Retrain", type="primary"):
    with st.spinner("Fetching latest telemetry for training subset..."):
        df = orch.fetch_and_process()
        time.sleep(1) # Simulated delay for UI feel
        
    if df is not None and len(df) > 50:
        with st.spinner("Retraining Model & Logging to MLflow..."):
            # Trigger the actual train function
            orch.engine.train(df[["distance_cm", "rolling_mean", "rolling_std"]])
            st.success("✅ Model retrained successfully! New version pushed to local registry.")
            st.balloons()
    else:
        st.error("❌ Insufficient data. Need at least 50 continuous telemetry packets.")