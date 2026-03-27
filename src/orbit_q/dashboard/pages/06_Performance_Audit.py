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
import pandas as pd
from firebase_admin import db
from orbit_q import config

st.set_page_config(page_title="Performance Audit", layout="wide")
st.title("🎯 AI Performance Audit")
st.markdown("Detailed breakdown of model accuracy and real-time anomaly rates.")

# Fetch system metrics
metrics = db.reference("/SYSTEM_METRICS").get() or {}
accuracy = metrics.get("accuracy", "N/A")
latency_gain = metrics.get("latency_gain", "N/A")

# Big KPI Row
col1, col2, col3 = st.columns(3)
col1.metric("Current Accuracy", accuracy, "Target: 95.00%")
col2.metric("Inference Optimization", latency_gain, "Target: 40.0%")
col3.metric("Configured Contamination", f"{config.CONTAMINATION * 100}%", "Expected Outlier Ratio")

st.divider()

# Calculate the *Actual* Anomaly Rate right now
st.subheader("Live Anomaly Distribution")
raw_data = db.reference("/SENSOR_DATA").order_by_key().limit_to_last(1000).get()
alerts = db.reference("/ML_ALERTS").order_by_key().limit_to_last(1000).get()

if raw_data and alerts:
    total_packets = len(raw_data)
    total_alerts = len(alerts)
    observed_rate = (total_alerts / total_packets) * 100

    c1, c2 = st.columns([1, 2])

    with c1:
        st.write(f"**Total Packets Evaluated:** {total_packets}")
        st.write(f"**Total Anomalies Flagged:** {total_alerts}")
        st.metric(
            "Observed Anomaly Rate",
            f"{observed_rate:.2f}%",
            delta=f"{(observed_rate - (config.CONTAMINATION * 100)):.2f}% vs Config",
            delta_color="inverse",
        )

    with c2:
        # Show a progress bar comparing observed vs expected
        st.write("**Expected vs Observed Anomaly Rate**")
        st.progress(config.CONTAMINATION, text=f"Expected ({config.CONTAMINATION * 100}%)")

        # Cap at 1.0 for progress bar safety
        safe_observed = min(observed_rate / 100, 1.0)
        st.progress(safe_observed, text=f"Observed ({observed_rate:.2f}%)")

        if observed_rate > (config.CONTAMINATION * 100) * 1.5:
            st.warning(
                "⚠️ Observed anomaly rate is significantly higher than configured contamination. Consider retraining the model."
            )
else:
    st.info("Gathering enough telemetry and alert data to calculate distribution...")
