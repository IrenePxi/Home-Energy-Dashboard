import streamlit as st
import pandas as pd
from datetime import time as _time
from services.scheduling import find_best_interval
from data_sources.electricity_prices import load_unified_price_data

def render_scheduling():
    with st.container(border=True):
        st.markdown("""
            <style>
            [data-testid="stMetricLabel"] { font-size: 13px !important; }
            [data-testid="stMetricValue"] { font-size: 20px !important; }
            .sched-device-name { font-size: 1.05rem !important; font-weight: 600; margin-bottom: -5px; margin-top: -10px; }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("#### 🧠 Smart Scheduling (Price Optimized)")
        st.caption("Automatically finds the cheapest time to run your selected devices.")

        # Predefined devices with default durations and power ratings (kW)
        DEVICE_SPECS = {
            "EV Charging (4h)": {"dur": 4.0, "power": 11.0},
            "Washing Machine (2h)": {"dur": 2.0, "power": 2.2},
            "Dishwasher (3h)": {"dur": 3.0, "power": 1.5},
            "Robot Vacuum (1h)": {"dur": 1.0, "power": 0.1}, 
            "Dryer (1.5h)": {"dur": 1.5, "power": 2.5},
            "Custom Device": {"dur": 1.0, "power": 1.0}
        }

        selected_devs = st.multiselect(
            "Select Devices to Schedule",
            list(DEVICE_SPECS.keys()),
            default=[],
            key="sched_multi_devs"
        )

        if not selected_devs:
            st.info("Select devices to see scheduling suggestions.")
            return

        # Automatic Window: Today 00:00 to Tomorrow 06:00
        today_start = pd.Timestamp.now().normalize()
        deadline = today_start + pd.Timedelta(days=1, hours=6)
        
        st.markdown(f"💡 *Optimizing for the window:* **Today 00:00** — **Tomorrow 06:00**")

        # Load Data Automatically
        # Load price data
        try:
            df_price = load_unified_price_data()
        except Exception as e:
            st.error(f"Error loading price data: {e}")
            return


        st.markdown("---")

        # Dialog Helper
        # Try to use st.dialog (1.34+) or fallback to experimental (1.33+)
        if hasattr(st, "dialog"):
            dialog_decorator = st.dialog
        elif hasattr(st, "experimental_dialog"):
            dialog_decorator = st.experimental_dialog
        else:
            # Fallback if neither exists (very old streamlit), use expander logic inline (not handled here)
            dialog_decorator = None
            st.warning("Update Streamlit to support dialogs.")

        if dialog_decorator:
            @dialog_decorator("Configure Device")
            def configure_device_dialog(dev_name):
                # Auto-pause live updates to prevent refresh closing the dialog
                if st.session_state.get("dt_live_toggle", True):
                    st.session_state["request_pause"] = True
                    st.rerun() 
                    
                # Read current values from state
                cur_dur = st.session_state.get(f"sched_dur_{dev_name}", DEVICE_SPECS[dev_name]["dur"])
                cur_pow = st.session_state.get(f"sched_pow_{dev_name}", DEVICE_SPECS[dev_name]["power"])
                cur_name = st.session_state.get(f"sched_name_{dev_name}", dev_name)
                
                st.write(f"Settings for **{cur_name}**")
                
                # Allow name editing for Custom Device (or all?)
                # User asked for freedom to type in name, so allow for all is fine, but essential for Custom.
                new_name = st.text_input("Name", value=cur_name)
                new_dur = st.number_input("Duration (h)", 0.5, 24.0, float(cur_dur), 0.5)
                new_power = st.number_input("Power (kW)", 0.1, 50.0, float(cur_pow), 0.1)
                
                # Resume updates on Apply
                if st.button("Apply"):
                    st.session_state[f"sched_dur_{dev_name}"] = new_dur
                    st.session_state[f"sched_pow_{dev_name}"] = new_power
                    st.session_state[f"sched_name_{dev_name}"] = new_name
                    st.session_state["request_resume"] = True
                    st.rerun()

        # Iterate and display
        for dev_name in selected_devs:
            specs = DEVICE_SPECS[dev_name]
            default_dur = specs["dur"]
            default_power = specs["power"]
            
            # Ensure state initialized
            if f"sched_dur_{dev_name}" not in st.session_state:
                st.session_state[f"sched_dur_{dev_name}"] = default_dur
            if f"sched_pow_{dev_name}" not in st.session_state:
                st.session_state[f"sched_pow_{dev_name}"] = default_power
            if f"sched_name_{dev_name}" not in st.session_state:
                 st.session_state[f"sched_name_{dev_name}"] = dev_name
                
            # Get active values
            active_dur = st.session_state[f"sched_dur_{dev_name}"]
            active_power = st.session_state[f"sched_pow_{dev_name}"]
            display_name = st.session_state[f"sched_name_{dev_name}"]
            
            # Calculate
            result = find_best_interval(
                df_price=df_price,
                duration_hours=active_dur,
                earliest_time=None, 
                latest_time=None,
                deadline_dt=deadline
            )
            
            # Render Card (Nested container for visual separation)
            with st.container(border=True):
                 # Header Row: Icon + Name + Settings Button
                 c_head1, c_head2 = st.columns([0.9, 0.1])
                 with c_head1:
                     tooltip_text = f"Rated Power: {active_power} kW\nDuration: {active_dur} h"
                     st.markdown(f'<p class="sched-device-name" title="{tooltip_text}">🔌 {display_name.split(" (")[0]}</p>', unsafe_allow_html=True)
                 with c_head2:
                     if st.button("⚙️", key=f"btn_set_{dev_name}"):
                         if dialog_decorator:
                             configure_device_dialog(dev_name)
                         else:
                             st.error("Dialogs not supported.")

                 if result:
                    start_str = result["start"].strftime("%A, %H:%M")
                    end_str = result["end"].strftime("%H:%M")
                    
                    # Calculate Totals
                    total_kwh = active_power * active_dur
                    total_cost = result['avg_price'] * total_kwh
                    
                    c_m1, c_m2, c_m3 = st.columns([1.5, 1, 1])
                    with c_m1:
                        st.metric("Best Time", f"{start_str} — {end_str}")
                    with c_m2:
                         st.metric("Consumed Energy", f"{total_kwh:.1f} kWh")
                    with c_m3:
                        st.metric("Total Cost", f"{total_cost:.2f} DKK")
                 else:
                      st.warning(f"Could not find schedule for {dev_name}")

