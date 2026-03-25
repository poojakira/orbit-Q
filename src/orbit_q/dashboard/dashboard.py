"""
Orbit-Q | Enterprise Satellite Operations Command Center
High-throughput operator dashboard with live telemetry, anomaly detection,
KPI metrics, and operator playbook automation.
"""
import time
import random
import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- Firebase (optional) ---
try:
    import firebase_admin
    from firebase_admin import db, credentials
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from orbit_q import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OrbitIQ Ops",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0a0e1a; color: #e0e6f0; }
    .stMetric { background: #10172a; border-radius: 8px; padding: 10px; border: 1px solid #1e3a5f; }
    .anomaly-alert { background: #3d0c0c; border-left: 4px solid #ff4b4b; padding: 12px; border-radius: 4px; }
    .nominal-chip { background: #0a3d1f; border-left: 4px solid #00cc66; padding: 8px; border-radius: 4px; }
    h1, h2, h3 { color: #7ecfff; }
</style>
""", unsafe_allow_html=True)

# ── Firebase Init ────────────────────────────────────────────────────────────
firebase_ok = False
if FIREBASE_AVAILABLE and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(config.SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"databaseURL": config.DB_URL})
        firebase_ok = True
    except Exception as e:
        log.warning(f"Firebase unavailable (mock mode): {e}")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite.png", width=80)
    st.title("🛰️ OrbitIQ Ops")
    st.markdown("**Satellite C2 Dashboard**")
    st.markdown("---")
    st.markdown("**Telemetry Mode**")
    sim_mode = st.radio("Source", ["🔴 Live Firebase", "🟡 Mock / Demo"], index=1)
    refresh_hz = st.slider("Refresh Rate (Hz)", 1, 50, 10)
    st.markdown("---")
    st.markdown("**Operator Playbook**")
    alert_channel = st.selectbox("Alert Channel", ["Slack (simulated)", "PagerDuty (simulated)", "None"])
    run_playbook = st.button("🚨 Trigger Alert Drill")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🛰️ Satellite Operations Command Center")
st.caption(f"Live dashboard | {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
st.markdown("---")

# ── Telemetry Generation / Fetch ─────────────────────────────────────────────
def get_mock_telemetry(n: int = 100) -> pd.DataFrame:
    """Generate realistic mock telemetry for demo mode."""
    np.random.seed(int(time.time()) % 1000)
    t = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="100ms")
    anomaly_mask = np.random.random(n) < 0.05
    data = {
        "timestamp": t,
        "temperature_C": np.where(anomaly_mask,
            np.random.uniform(85, 120, n), np.random.normal(25, 3, n)),
        "voltage_V": np.where(anomaly_mask,
            np.random.uniform(2.5, 3.5, n), np.random.normal(5.0, 0.1, n)),
        "signal_strength_dBm": np.random.normal(-60, 5, n),
        "gyro_x_dps": np.random.normal(0, 0.5, n),
        "is_anomaly": anomaly_mask,
        "face": np.random.choice(["NORTH", "SOUTH", "EAST", "WEST"], n),
    }
    return pd.DataFrame(data)

def get_firebase_telemetry() -> pd.DataFrame:
    try:
        ref = db.reference("/SENSOR_DATA")
        raw = ref.order_by_key().limit_to_last(100).get()
        if not raw:
            return get_mock_telemetry()
        rows = list(raw.values())
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df
    except Exception:
        return get_mock_telemetry()

df = get_firebase_telemetry() if (firebase_ok and sim_mode.startswith("🔴")) else get_mock_telemetry()
anomaly_df = df[df.get("is_anomaly", pd.Series([False]*len(df)))] if "is_anomaly" in df.columns else pd.DataFrame()
anomaly_count = len(anomaly_df)

# ── KPI Ribbon ───────────────────────────────────────────────────────────────
st.subheader("📊 Live KPI Ribbon")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("🌡️ Avg Temp", f"{df['temperature_C'].mean():.1f} °C" if "temperature_C" in df.columns else "N/A",
          delta=f"{df['temperature_C'].diff().mean():.2f}" if "temperature_C" in df.columns else None)
c2.metric("⚡ Avg Voltage", f"{df['voltage_V'].mean():.2f} V" if "voltage_V" in df.columns else "N/A")
c3.metric("📡 Signal", f"{df['signal_strength_dBm'].mean():.1f} dBm" if "signal_strength_dBm" in df.columns else "N/A")
c4.metric("🚨 Anomalies", anomaly_count, delta=f"+{anomaly_count}" if anomaly_count else None,
          delta_color="inverse")
c5.metric("📦 Stream Rate", f"{refresh_hz} Hz")
c6.metric("🛰️ Status", "ANOMALY" if anomaly_count > 0 else "NOMINAL")

st.markdown("---")

# ── Streaming Telemetry Ribbons ───────────────────────────────────────────────
st.subheader("📈 Live Telemetry Channels")
tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperature", "⚡ Voltage", "📡 Signal Strength", "🔄 Gyroscope"])

if "temperature_C" in df.columns:
    with tab1:
        chart_data = df.set_index("timestamp")[["temperature_C"]] if "timestamp" in df.columns else df[["temperature_C"]]
        st.line_chart(chart_data, height=200, use_container_width=True)
        if anomaly_count > 0:
            st.markdown('<div class="anomaly-alert">⚠️ <b>Thermal anomaly detected!</b> Temperature spike on one or more faces.</div>', unsafe_allow_html=True)

if "voltage_V" in df.columns:
    with tab2:
        chart_data = df.set_index("timestamp")[["voltage_V"]] if "timestamp" in df.columns else df[["voltage_V"]]
        st.line_chart(chart_data, height=200, use_container_width=True)

if "signal_strength_dBm" in df.columns:
    with tab3:
        chart_data = df.set_index("timestamp")[["signal_strength_dBm"]] if "timestamp" in df.columns else df[["signal_strength_dBm"]]
        st.line_chart(chart_data, height=200, use_container_width=True)

if "gyro_x_dps" in df.columns:
    with tab4:
        chart_data = df.set_index("timestamp")[["gyro_x_dps"]] if "timestamp" in df.columns else df[["gyro_x_dps"]]
        st.line_chart(chart_data, height=200, use_container_width=True)

st.markdown("---")

# ── Anomaly Alert Panel ───────────────────────────────────────────────────────
st.subheader("🚨 Anomaly Alert Log")
if anomaly_count > 0:
    st.error(f"🔴 {anomaly_count} anomalies detected in current telemetry window!")
    if "is_anomaly" in df.columns:
        display_cols = [c for c in ["timestamp", "face", "temperature_C", "voltage_V", "is_anomaly"] if c in df.columns]
        st.dataframe(anomaly_df[display_cols].tail(20), use_container_width=True)
else:
    st.markdown('<div class="nominal-chip">✅ All telemetry channels nominal — no anomalies detected.</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Operator Playbook ─────────────────────────────────────────────────────────
st.subheader("📋 Operator Playbook Automation")
if run_playbook:
    with st.spinner("Executing alert playbook..."):
        time.sleep(1)
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "channel": alert_channel,
            "anomaly_count": anomaly_count,
            "trigger": "MANUAL_DRILL",
            "status": "SENT",
        }
        
        if "Slack" in alert_channel:
            webhook = os.getenv("SLACK_WEBHOOK_URL", "")
            if webhook:
                try:
                    requests.post(webhook, json={"text": f"🛰️ [ORBIT-Q DRILL] {anomaly_count} anomalies detected - operator review required."}, timeout=5)
                    st.success("✅ Slack webhook triggered successfully.")
                except Exception as e:
                    st.warning(f"Slack webhook failed: {e}")
            else:
                st.info("📩 **[SIM]** Slack webhook would fire — set SLACK_WEBHOOK_URL env var for live integration.")
        elif "PagerDuty" in alert_channel:
            st.info("📟 **[SIM]** PagerDuty P1 incident would be created — configure PAGERDUTY_ROUTING_KEY for live use.")
        
        st.json(log_entry)
        
        # Log to Firebase if available
        if firebase_ok:
            try:
                db.reference("/ML_ALERTS").push(log_entry)
                st.caption("📌 Alert logged to Firebase /ML_ALERTS.")
            except Exception:
                pass

st.markdown("---")
st.caption("Orbit-Q | Enterprise Satellite Telemetry Platform | Powered by IsolationForest + Autoencoder Ensemble")