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
import plotly.express as px

st.set_page_config(page_title="Orbital Tracking", layout="wide")
st.title("🛰️ Orbital Sensor Radar")
st.markdown("Live 360° view of spatial proximity across all satellite faces.")

# Fetch the latest 100 records to ensure we get data for all 4 faces
raw_data = db.reference("/SENSOR_DATA").order_by_key().limit_to_last(100).get()

if raw_data:
    df = pd.DataFrame(raw_data.values())
    
    # Get the absolute latest reading for each face
    latest_readings = df.sort_values('timestamp').groupby('face').tail(1)
    
    # Fill in any missing faces just in case
    faces = ["NORTH", "EAST", "SOUTH", "WEST"] # Ordered clockwise
    radar_data = []
    for face in faces:
        match = latest_readings[latest_readings['face'] == face]
        dist = match['distance_cm'].values[0] if not match.empty else 0
        radar_data.append({"face": face, "distance_cm": dist})
        
    df_radar = pd.DataFrame(radar_data)

    # Build the Radar Chart
    fig = px.line_polar(df_radar, r='distance_cm', theta='face', line_close=True,
                        template="plotly_dark", markers=True, 
                        title="Live Proximity Radar (cm)")
    
    fig.update_traces(fill='toself', line_color='#00a4e4')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Operator Tip:** If the radar spikes heavily in one direction (e.g., >300cm), the system will automatically flag it as an anomaly on the Alert Command page.")
else:
    st.error("Awaiting telemetry stream to render radar...")