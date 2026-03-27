# Inside your Streamlit page (mlops_lineage.py)
import streamlit as st
import mlflow


def display_metrics():
    # Fetch the latest run from MLflow
    latest_run = mlflow.search_runs(order_by=["start_time DESC"]).iloc[0]

    accuracy = latest_run["metrics.accuracy"]  # Assuming you logged this
    latency_imp = latest_run["metrics.latency_improvement_pct"]

    # Create visual Gauges or Metrics in Mission Control
    col1, col2 = st.columns(2)
    col1.metric("Model Predictive Accuracy", f"{accuracy:.1f}%", "+0.2%")
    col2.metric("Response Time Improvement", f"{latency_imp:.0f}%", "Target: >40%")


st.header("🛰️ MLOps Mission Health")
display_metrics()
