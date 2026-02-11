import sys
import os
from pathlib import Path
import streamlit as st

# Fix import path for DT_dashboard module
# Add the project root (parent of DT_dashboard) to sys.path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import DT_dashboard.ui.tabs.forecasts as kv_forecasts
import DT_dashboard.ui.tabs.dt_real as kv_dt_real

st.set_page_config(
    page_title="Home Energy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.showSidebarNavigation", True)

st.title("🏠 Home Energy Dashboard")

# Initialize MQTT Connection (Backend)
if "mqtt_client" not in st.session_state:
    from DT_dashboard.services.mqtt_manager import get_client
    get_client()  # Starts the loop and stores in session_state

# Main Dashboard Layout
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

# Row 2: CO2 + Weather + Scheduling
r2_col1, r2_col2, r2_col3 = st.columns(3, gap="medium")

with r2_col1:
    kv_forecasts.render_co2_forecast()

with r2_col2:
    kv_forecasts.render_weather_forecast()

with r2_col3:
    # Import here to avoid circular dependencies if any
    import DT_dashboard.ui.tabs.scheduling_ui as kv_scheduling
    kv_scheduling.render_scheduling()
