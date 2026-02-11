import json
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from services.paths import rpath
from services.mqtt_manager import start_mqtt_listener, get_latest_data, get_connection_status, get_last_message

def render_dt_real():
    # Start the background MQTT listener (singleton)
    start_mqtt_listener()
    
    # Auto-refresh every 2 seconds to fetch new data from the listener cache
    # Check if a long-running task is active (e.g. Price/PV update) to prevent interruption
    
    # Allow pausing live updates (e.g. when configuring scheduling)
    # Handle requests from other modules to change toggle state
    if st.session_state.get("request_pause"):
        st.session_state["dt_live_toggle"] = False
        del st.session_state["request_pause"]
    if st.session_state.get("request_resume"):
        st.session_state["dt_live_toggle"] = True
        del st.session_state["request_resume"]

    st.markdown("#### 🏠 Real-time Power Flow")
    live = st.toggle("Live Updates", value=True, key="dt_live_toggle")

    if live and not st.session_state.get("long_running_task", False):
        st_autorefresh(interval=2000, key="dashboard_refresh")

    data = get_latest_data()
    status = get_connection_status()
    
    # Minimal UI
    if status == "Connected":
        st.success(f"**Status:** {status} | ⚡ Live Power Flow")
    else:
        st.error(f"**Status:** {status}")

    try:
        with open(rpath("dashboard.html"), "r", encoding="utf-8") as f:
            html_template = f.read()
            
        # Inject data into the HTML as a global JS variable
        # We replace the placeholder or inject it before the script
        json_data = json.dumps(data)
        injection = f"<script>window.serverData = {json_data};</script>"
        
        # Inject before </head> or <body>
        html_content = html_template.replace("</head>", f"{injection}</head>")
            
        # Ensure height is sufficient
        components.html(html_content, height=350, scrolling=False)
            
    except FileNotFoundError as e:
        st.error(f"Dashboard HTML file not found: {e}")
