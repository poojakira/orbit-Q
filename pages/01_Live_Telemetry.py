# --- AUTO-ADDED FIREBASE INIT ---
import firebase_admin
from firebase_admin import credentials, db
import config

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(config.SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"databaseURL": config.DB_URL})
except Exception:
    pass
# --------------------------------

import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Live Telemetry", layout="wide")

st.title("📈 Real-Time Telemetry Stream")
st.markdown("Live 2-second polling of satellite sensor distances.")

placeholder = st.empty()

# Create a continuous loop to stream data into the UI
while True:
    try:
        data = db.reference("/SENSOR_DATA").order_by_key().limit_to_last(100).get()
        if data:
            # Convert to DataFrame and transpose
            df = pd.DataFrame(data).T
            
            # Convert unix timestamp to readable datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            with placeholder.container():
                # Plot the distance metric over time
                st.line_chart(df.set_index('timestamp')['distance_cm'])
                
                # THE FIX: Use st.table() instead of st.dataframe() to prevent React Error #185
                st.table(df.tail(5))
                
    except Exception as e:
        st.warning(f"Awaiting telemetry connection... (Error: {e})")
    
    # Wait 2 seconds before pulling the next batch of data
    time.sleep(2)