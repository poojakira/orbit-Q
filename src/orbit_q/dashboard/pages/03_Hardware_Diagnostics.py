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
import plotly.graph_objects as go
import random

st.set_page_config(page_title="Hardware Diagnostics", layout="wide")
st.title("⚙️ Satellite Hardware Diagnostics")

col1, col2, col3 = st.columns(3)


def plot_gauge(title, val, max_val, suffix=""):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": title},
            gauge={"axis": {"range": [None, max_val]}, "bar": {"color": "#00a4e4"}},
        )
    )
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
    return fig


with col1:
    st.plotly_chart(plot_gauge("CPU Load", random.randint(30, 60), 100, "%"), use_container_width=True)
with col2:
    st.plotly_chart(plot_gauge("Internal Temp", random.uniform(15.0, 25.0), 50, "°C"), use_container_width=True)
with col3:
    st.plotly_chart(plot_gauge("Solar Array Output", random.uniform(85.0, 99.0), 100, "W"), use_container_width=True)

st.divider()
st.subheader("System Event Log")
st.code(
    """
[2026-02-23 18:00:01] SYSTEM: Solar array alignment adjusted.
[2026-02-23 18:05:12] MLOPS: Inference engine memory cleared.
[2026-02-23 18:10:44] SENSOR: Nominal ping from all faces (NORTH, SOUTH, EAST, WEST).
""",
    language="text",
)
