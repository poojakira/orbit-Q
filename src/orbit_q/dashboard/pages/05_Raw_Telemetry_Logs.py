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

st.set_page_config(page_title="Raw Telemetry", layout="wide")
st.title("🗄️ Raw Telemetry Logs")
st.markdown("Explore and export raw sensor packets for manual offline analysis.")

raw_data = db.reference("/SENSOR_DATA").order_by_key().limit_to_last(1000).get()

if raw_data:
    df = pd.DataFrame(raw_data.values())
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp', ascending=False)

    # Interactive Filters
    col1, col2 = st.columns(2)
    with col1:
        face_filter = st.multiselect("Filter by Face", options=df['face'].unique(), default=df['face'].unique())
    with col2:
        dist_filter = st.slider("Filter by Distance (cm)", min_value=float(df['distance_cm'].min()), 
                                max_value=float(df['distance_cm'].max()), 
                                value=(float(df['distance_cm'].min()), float(df['distance_cm'].max())))

    # Apply Filters
    filtered_df = df[
        (df['face'].isin(face_filter)) & 
        (df['distance_cm'] >= dist_filter[0]) & 
        (df['distance_cm'] <= dist_filter[1])
    ]

    st.dataframe(filtered_df, use_container_width=True, height=400)

    # Enterprise Feature: CSV Export
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='orbit_iq_telemetry.csv',
        mime='text/csv',
    )
else:
    st.error("No telemetry data found in Firebase.")