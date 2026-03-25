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
import mlflow
import pandas as pd
from orbit_q import config

st.set_page_config(page_title="MLflow Lineage", layout="wide")
st.title("📜 MLflow Experiment Lineage")
st.markdown("Audit log of all model training runs and inference performance pulses.")

# Force Streamlit to look at the local mlruns folder we configured in orchestrator
mlflow.set_tracking_uri(config.MLFLOW_URI)
mlflow.set_experiment(config.EXPERIMENT_NAME)

try:
    # Fetch all runs from MLflow
    runs = mlflow.search_runs(order_by=["start_time DESC"])
    
    if not runs.empty:
        # Clean up the dataframe for the dashboard
        display_df = runs[['run_id', 'start_time', 'tags.mlflow.runName', 'metrics.model_accuracy_pct', 'metrics.response_time_improvement_pct']]
        display_df.columns = ['Run ID', 'Timestamp', 'Run Name', 'Accuracy (%)', 'Latency Gain (%)']
        
        st.dataframe(display_df.style.highlight_max(subset=['Accuracy (%)'], color='lightgreen'), 
                     use_container_width=True, height=400)
                     
        st.info("👆 The highlighted row represents the highest performing model configuration.")
    else:
        st.warning("No MLflow runs found. Make sure the orchestrator is running.")
except Exception as e:
    st.error(f"Could not connect to MLflow local storage. Error: {e}")