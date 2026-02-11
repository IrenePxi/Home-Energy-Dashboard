import sys
import os
from pathlib import Path
import streamlit as st

import ui.tabs.forecasts as kv_forecasts
import ui.tabs.dt_real as kv_dt_real

st.set_page_config(
    page_title="Home Energy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.showSidebarNavigation", True)

st.title("🏠 Home Energy Dashboard")

# Initialize MQTT Connection (Backend)
if "mqtt_client" not in st.session_state:
    from services.mqtt_manager import get_client
    get_client()  # Starts the loop and stores in session_state

# Fetch shared data once at top level
try:
    from data_sources.co2 import fetch_co2_prog
    df_co2_shared = fetch_co2_prog(area="DK1", horizon_hours=96)
except Exception:
    df_co2_shared = None

# Main Dashboard Layout
# Row 1: Real-Time House + EL Price + PV
# Structure: House (wide) | Price | PV
r1_col1, r1_col2, r1_col3 = st.columns([1, 1, 1], gap="medium")

with r1_col1:
    with st.container(border=True):
        kv_dt_real.render_dt_real()

with r1_col2:
    kv_forecasts.render_electricity_price()

with r1_col3:
    kv_forecasts.render_pv_forecast()
    kv_forecasts.render_gas_price()

# Row 2: CO2 + Weather + Scheduling
r2_col1, r2_col2, r2_col3 = st.columns(3, gap="medium")

with r2_col1:
    kv_forecasts.render_co2_forecast(df_co2=df_co2_shared)

with r2_col2:
    kv_forecasts.render_weather_forecast()

with r2_col3:
    # Import here to avoid circular dependencies if any
    import ui.tabs.scheduling_ui as kv_scheduling
    kv_scheduling.render_scheduling(df_co2=df_co2_shared)
