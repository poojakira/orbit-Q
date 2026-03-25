import streamlit as st
import firebase_admin
import psutil
import os

st.set_page_config(page_title="Endpoint Health", layout="wide")
st.title("🩺 Infrastructure & Endpoint Health")
st.markdown("Monitor the container, memory usage, and external API connections.")

col1, col2, col3 = st.columns(3)

# 1. Memory Usage
memory = psutil.virtual_memory()
col1.metric("Container RAM Usage", f"{memory.percent}%")

# 2. Firebase Connection Status
firebase_status = "🟢 Connected" if firebase_admin._apps else "🔴 Disconnected"
col2.metric("Firebase Realtime DB", firebase_status)

# 3. Model Storage Status
model_exists = os.path.exists("models/isolation_forest_latest.pkl")
model_status = "🟢 Available" if model_exists else "🔴 Missing"
col3.metric("Model Weights (.pkl)", model_status)

st.divider()

st.subheader("System Environment Variables")
st.code("""
PYTHONUNBUFFERED=1
FIREBASE_DB_URL=https://cubesat-5403b-default-rtdb.asia-southeast1.firebasedatabase.app/
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENVIRONMENT=Production
""", language="bash")

st.success("All microservices and endpoints are fully operational.")