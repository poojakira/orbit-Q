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
import time

st.set_page_config(page_title="Alert Command", layout="wide")
st.title("🚨 Critical Alert Command")
st.markdown("Monitor AI-flagged space debris and sensor anomalies.")

alerts = db.reference("/ML_ALERTS").order_by_key().limit_to_last(50).get()

if alerts:
    # Convert Firebase dictionary to DataFrame
    df_alerts = pd.DataFrame(alerts.values())
    df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])
    df_alerts = df_alerts.sort_values(by="timestamp", ascending=False)

    # Highlight high-risk anomalies
    def color_risk(val):
        color = "#ff4b4b" if val == -1 else "white"
        return f"color: {color}"

    st.dataframe(df_alerts.style.map(color_risk, subset=["anomaly"]), use_container_width=True, height=500)
else:
    st.success("✅ No critical anomalies detected in the current mission window.")
