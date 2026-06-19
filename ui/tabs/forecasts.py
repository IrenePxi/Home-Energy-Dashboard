import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from services.scripts import run_py
from data_sources.electricity_prices import load_electricity_prices, load_unified_price_data
from data_sources.pv import load_pv_predictions
from data_sources.gas import fetch_gas_prices
from services.prediction_refresh import clear_auto_refresh_failure, get_auto_refresh_error

_TZ = "Europe/Copenhagen"
def _now_dk():
    """Current wall-clock time in Europe/Copenhagen, returned tz-naive for chart use."""
    return pd.Timestamp.now(tz=_TZ).replace(tzinfo=None)

_DAY_COLORS = ("rgba(230,240,255,0.55)", "rgba(255,243,224,0.55)")

def _add_day_bands(fig, x_min, x_max, colors=_DAY_COLORS):
    """Add alternating day-colored vertical bands behind the traces."""
    try:
        x_min, x_max = pd.Timestamp(x_min), pd.Timestamp(x_max)
    except Exception:
        return
    day0 = x_min.normalize()
    day_end = x_max.normalize() + pd.Timedelta(days=1)
    days = pd.date_range(day0, day_end, freq="D")
    for i in range(len(days) - 1):
        fig.add_vrect(
            x0=days[i], x1=days[i + 1],
            fillcolor=colors[i % len(colors)],
            opacity=1.0, line_width=0, layer="below",
        )

# Callbacks to lock autorefresh
def start_price_update():
    st.session_state["updating_price"] = True
    st.session_state.pop("price_auto_updated", None)
    clear_auto_refresh_failure("electricity_price")

def start_pv_update():
    st.session_state["updating_pv"] = True
    st.session_state.pop("pv_auto_updated", None)
    st.session_state.pop("pv_update_msg", None)  # Clear previous result
    clear_auto_refresh_failure("pv_forecast")

def render_electricity_price():
    # --- 1. Electricity Price Prediction Block ---
    with st.container(border=True):
        st.markdown("#### 💡 Electricity Price Prediction")
        
        # --- Accuracy Metrics Display ---
        from services.paths import results_dir
        metrics_file = results_dir() / "prediction_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    m = json.load(f)
                
                # Create 3 columns for metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE (Accuracy)", f"{m['mae']} DKK")
                c2.metric("RMSE", f"{m['rmse']}")
                c3.metric("MAPE", f"{m['mape_pct']}%")
                st.caption(f"Last validated on {m['last_updated']} (held-out 24h test)")
            except:
                pass

        # Display Price Charts

        try:
            # Loaded from top-level import
            df_price = load_unified_price_data()
            refresh_err = get_auto_refresh_error("electricity_price")
            if refresh_err:
                st.warning(f"Electricity predictions could not be refreshed automatically: {refresh_err}")

            # Filter to today and tomorrow
            today_start = _now_dk().normalize()
            tomorrow_end = today_start + pd.Timedelta(days=2)
            df_hourly_view = df_price[
                (df_price["DateTime"] >= today_start) & 
                (df_price["DateTime"] < tomorrow_end) &
                (df_price["Source"] == "Actual")
            ].copy()
            
            if not df_hourly_view.empty:
                # Bar chart logic
                df_bar = df_hourly_view.set_index("DateTime")[["SpotPrice_DKK_per_kWh"]].resample("h").mean().reset_index()
                
                # --- Auto Update Logic for El Price ---
                now_dk = _now_dk()
                
                # Helper to check file age
                from services.paths import results_dir
                price_file = results_dir() / "Electricity_price_prediction_result.csv"
                file_age_hours = 999
                if price_file.exists():
                    file_age_hours = (now_dk.timestamp() - price_file.stat().st_mtime) / 3600

                # Determine if we need an update
                latest_act = df_price[df_price["Source"] == "Actual"]["DateTime"].max() if not df_price.empty else pd.Timestamp(0)
                latest_pred = df_price[df_price["Source"] == "Predicted"]["DateTime"].max() if not df_price.empty else pd.Timestamp(0)
                
                # Check for "Forward Leap": if actual data exists for tomorrow, but prediction was made when we only had today's data
                # Or if the prediction doesn't cover "Tomorrow + 1 day"
                # Or if the file is more than 4 hours old
                needs_update = False
                if latest_pred < (now_dk.normalize() + pd.Timedelta(days=2)):
                    needs_update = True
                elif file_age_hours > 4:
                    needs_update = True
                elif latest_act > (now_dk.normalize() + pd.Timedelta(hours=12)): # We have tomorrow's prices (DK price release)
                    # If the file hasn't been updated since those prices were released
                    pass # We trust file_age or tomorrow_coverage mostly

                if not st.session_state.get("updating_price", False) and not st.session_state.get("price_auto_updated", False):
                    if needs_update and not get_auto_refresh_error("electricity_price"):
                        st.session_state["updating_price"] = True
                        st.rerun()

                threshold = df_bar["SpotPrice_DKK_per_kWh"].quantile(0.75)
                df_bar["Color"] = df_bar["SpotPrice_DKK_per_kWh"].apply(lambda x: "#FFD700" if x >= threshold else "#1f77b4")
                
                fig_bar = go.Figure(data=[go.Bar(
                    x=df_bar["DateTime"], y=df_bar["SpotPrice_DKK_per_kWh"],
                    marker_color=df_bar["Color"], name="Hourly Price",
                    hovertemplate="%{x|%H:%M}<br>%{y:.4f} DKK/kWh<extra></extra>"
                )])
                
                # Legend items
                fig_bar.add_trace(go.Bar(x=[None], y=[None], marker_color="#FFD700", name="High Price Hours"))
                fig_bar.add_trace(go.Bar(x=[None], y=[None], marker_color="#1f77b4", name="Normal Price Hours"))
                
                latest_dt = df_bar["DateTime"].max()
                
                now = _now_dk().round("1min")
                fig_bar.add_vline(x=now, line_width=2, line_dash="dash", line_color="red")
                
                fig_bar.update_layout(
                    title="Today's Hourly Prices",
                    yaxis_title="Spot Price (DKK/kWh)",
                    showlegend=False,
                    xaxis_tickformat="%a %H:%M",
                    margin=dict(l=20, r=20, t=40, b=20),
                    uirevision="chart_state"
                )
                st.plotly_chart(fig_bar, width='stretch', height=250, key="chart_price_bar")
                
                # Logic for running the Price ML script
                if st.session_state.get("updating_price", False):
                    st.session_state["long_running_task"] = True
                    with st.spinner("Running El Price prediction script..."):
                        try:
                            from data_sources.electricity_prices import update_electricity_predictions
                            from services.mqtt_publisher import publish_electricity_price
                            
                            df_res = update_electricity_predictions()
                            if not df_res.empty:
                                if publish_electricity_price(df_res):
                                    st.success("Done! Data published.")
                                else:
                                    st.warning("Prediction done, but MQTT publish failed.")
                            else:
                                if "updating_price" in st.session_state:
                                    st.error("Prediction failed.")
                        except Exception as e:
                            st.error(f"Failed: {e}")
                        finally:
                            st.session_state["updating_price"] = False
                            st.session_state["long_running_task"] = False
                            st.session_state["price_auto_updated"] = True
                            clear_auto_refresh_failure("electricity_price")
                            st.rerun()
            else:
                st.info("No hourly price data available for today.")
                st.caption("⚠️ Data source unavailable — energidataservice.dk could not be reached. Prices will appear once the connection is restored.")

                # If no data at all, maybe try auto-update once
                if not st.session_state.get("updating_price", False) and not st.session_state.get("price_auto_updated", False):
                    if not get_auto_refresh_error("electricity_price"):
                        st.session_state["updating_price"] = True
                        st.rerun()

            # Line Chart

            if not df_price.empty:
                start_plot = _now_dk() - pd.Timedelta(days=2)
                end_plot = _now_dk() + pd.Timedelta(days=4)
                df_plot = df_price[(df_price["DateTime"] >= start_plot) & (df_price["DateTime"] <= end_plot)].copy()
                
                fig = go.Figure()
                
                # Actual
                df_act = df_plot[df_plot["Source"] == "Actual"]
                if not df_act.empty:
                    fig.add_trace(go.Scatter(
                        x=df_act["DateTime"], y=df_act["SpotPrice_DKK_per_kWh"],
                        mode='lines', name='Actual', line=dict(color='#1f77b4', width=2),
                        fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)'
                    ))
                
                # Predicted
                df_pred = df_plot[df_plot["Source"] == "Predicted"]
                if not df_pred.empty:
                    fig.add_trace(go.Scatter(
                        x=df_pred["DateTime"], y=df_pred["SpotPrice_DKK_per_kWh"],
                        mode='lines', name='Predicted', line=dict(color='#ff7f0e', width=2, dash='dash'),
                        fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.1)'
                    ))
                
                # Now line
                now = _now_dk().round("1min")
                fig.add_shape(type="line", x0=now, y0=0, x1=now, y1=1, xref="x", yref="paper", line=dict(color="red", width=2, dash="dot"))
                fig.add_annotation(x=now, y=1.05, xref="x", yref="paper", text="Now", showarrow=False, font=dict(color="red"))

                fig.update_layout(
                    title="Price Forecast Trend & Accuracy Analysis",
                    yaxis_title="DKK/kWh",
                    hovermode="x unified",
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=40, b=20),
                    uirevision="chart_state"
                )
                price_xmin = df_plot["DateTime"].min()
                price_xmax = df_plot["DateTime"].max()
                _add_day_bands(fig, price_xmin, price_xmax)
                fig.update_xaxes(
                    type="date", showgrid=True, griddash="dot",
                    dtick=12 * 3600 * 1000,
                    tick0=_now_dk().normalize(),
                    tickformat="%H:%M\n%a %b %d",
                    range=[price_xmin, price_xmax],
                )
                st.plotly_chart(fig, width='stretch', height=250, key="chart_price_trend")

        except Exception as e:
            import requests
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                st.warning("⚠️ Data source unavailable — energidataservice.dk could not be reached.")
            else:
                st.error(f"Error loading price data: {e}")

def render_pv_forecast():
    # --- 2. PV Power Prediction Block ---
    with st.container(border=True):
        st.markdown('#### ☀️ PV Power Prediction <span style="color:#666; font-size:1rem; font-weight:400;">(16 panels × 400W)</span>', unsafe_allow_html=True)
        
        if st.session_state.get("updating_pv", False):
            st.session_state["long_running_task"] = True
            with st.spinner("Running PV prediction script..."):
                try:
                    from data_sources.pv import run_pv_prediction
                    from services.mqtt_publisher import publish_pv_forecast
                    
                    df_res = run_pv_prediction()
                    # run_pv_prediction returns False on failure, or a DataFrame on success
                    if df_res is False or df_res is None:
                        st.session_state["pv_update_msg"] = ("error", "Prediction failed — check that PV data files exist and weather API is reachable.")
                    elif hasattr(df_res, 'empty') and not df_res.empty:
                        if publish_pv_forecast(df_res):
                            st.session_state["pv_update_msg"] = ("success", "Done! PV prediction updated and published.")
                        else:
                            st.session_state["pv_update_msg"] = ("warning", "Prediction complete, but MQTT publish failed (broker may not be reachable from this environment).")
                    else:
                        st.session_state["pv_update_msg"] = ("error", "Prediction returned empty results. Check logs.")
                except Exception as e:
                    st.session_state["pv_update_msg"] = ("error", f"Failed: {e}")
                finally:
                    st.session_state["updating_pv"] = False
                    st.session_state["long_running_task"] = False
                    st.rerun()

        # Show persistent result message from last update run
        pv_msg = st.session_state.get("pv_update_msg", None)
        if pv_msg:
            level, text = pv_msg
            if level == "success": st.success(text)
            elif level == "warning": st.warning(text)
            else: st.error(text)

        try:
            df_pv = load_pv_predictions()
            refresh_err = get_auto_refresh_error("pv_forecast")
            if refresh_err:
                st.warning(f"PV predictions could not be refreshed automatically: {refresh_err}")

            # Check if prediction data is stale or file is old
            latest_pv_dt = pd.to_datetime(df_pv["DateTime"]).max()
            today_start = _now_dk().normalize()
            
            from services.paths import results_dir
            pv_file = results_dir() / "pv_prediction_result.csv"
            pv_file_age_hours = (pd.Timestamp.now().timestamp() - pv_file.stat().st_mtime) / 3600 if pv_file.exists() else 999

            if latest_pv_dt < (today_start + pd.Timedelta(days=2)) or pv_file_age_hours > 4:
                if latest_pv_dt < today_start:
                    days_stale = (today_start - latest_pv_dt.normalize()).days
                    st.warning(f"⚠️ PV prediction is outdated (last updated {days_stale} day(s) ago). refreshing automatically...")
                
                # --- Auto Update Logic for PV ---
                if not st.session_state.get("updating_pv", False) and not st.session_state.get("pv_auto_updated", False):
                    if not get_auto_refresh_error("pv_forecast"):
                        st.session_state["updating_pv"] = True
                        st.rerun()

            fig_pv = px.line(df_pv, x="DateTime", y="Corrected_PV", title="Predicted PV Power", labels={"DateTime": "Time", "Corrected_PV": "PV Power (kW)"})
            now = _now_dk().round("1min")
            fig_pv.add_vline(x=now, line_width=2, line_dash="dash", line_color="red")
            
            pv_xmin, pv_xmax = df_pv["DateTime"].min(), df_pv["DateTime"].max()
            _add_day_bands(fig_pv, pv_xmin, pv_xmax)
            fig_pv.update_xaxes(
                type="date", showgrid=True, griddash="dot",
                dtick=12 * 3600 * 1000,
                tick0=_now_dk().normalize(),
                tickformat="%H:%M\n%a %b %d",
                range=[pv_xmin, pv_xmax],
            )
            fig_pv.update_layout(
                hovermode="x unified", 
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20),
                uirevision="chart_state"
            )
            st.plotly_chart(fig_pv, width='stretch', height=180, key="chart_pv_forecast")
        except Exception as e:
            st.warning(f"Could not load PV predictions: {e}")

def render_weather_forecast():
    # --- 3. Weather Forecast Block ---
    with st.container(border=True):
        st.markdown("#### 🌡️ Weather Forecast")
        
        try:
            from data_sources.weather import fetch_weather_open_meteo
            LAT, LON = 57.048, 9.921
            now = _now_dk().round("1min")
            start_date = (now - pd.Timedelta(days=1)).normalize()
            end_date = (now + pd.Timedelta(days=4)).normalize()
            
            with st.spinner("Fetching weather data..."):
                df_weather = fetch_weather_open_meteo(LAT, LON, start_date, end_date)
            
            if not df_weather.empty and "temp" in df_weather.columns:
                fig_temp = px.line(
                    df_weather.reset_index(), x="time", y="temp",
                    title="Temperature Forecast (°C)",
                    labels={"time": "Time", "temp": "Temperature (°C)"},
                    color_discrete_sequence=["#ff7f0e"]
                )
                fig_temp.add_vline(x=now, line_width=2, line_dash="dash", line_color="red", opacity=0.7)
                fig_temp.add_annotation(x=now, y=df_weather["temp"].max(), text="Now", showarrow=False, yshift=10)
                
                fig_temp.update_layout(
                    hovermode="x unified", xaxis_rangeslider_visible=False,
                    yaxis_title="Temperature (°C)", showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    uirevision="chart_state"
                )
                wx_xmin = df_weather.reset_index()["time"].min()
                wx_xmax = df_weather.reset_index()["time"].max()
                _add_day_bands(fig_temp, wx_xmin, wx_xmax)
                fig_temp.update_xaxes(
                    showgrid=True, griddash="dot",
                    dtick=12 * 3600 * 1000,
                    tick0=_now_dk().normalize(),
                    tickformat="%H:%M\n%a %b %d",
                    range=[wx_xmin, wx_xmax],
                )
                fig_temp.update_yaxes(showgrid=True, griddash="dot")
                st.plotly_chart(fig_temp, width='stretch', height=250, key="chart_weather_temp")
            else:
                st.warning("⚠️ Weather data temporarily unavailable — open-meteo.com may be rate limiting requests. Data is cached for 2 hours; it will retry automatically on the next cache refresh.")
                
        except Exception as e:
            st.warning("⚠️ Weather data temporarily unavailable. Will retry automatically.")

def render_co2_forecast(df_co2=None):
    # --- 4. CO2 Emission Forecast Block ---
    with st.container(border=True):
        st.markdown("#### 🌍 CO₂ Forecast")
        
        try:
            # Use shared data if available, otherwise fetch locally
            if df_co2 is None:
                from data_sources.co2 import fetch_co2_prog
                # Fetch data (horizon=96 to catch full future + today's history if available)
                with st.spinner("Fetching CO2 data..."):
                    df_co2 = fetch_co2_prog(area="DK1", horizon_hours=96)
            
            # Filter to start from today 00:00
            if df_co2 is not None and not df_co2.empty:
                # Essential copy to avoid modifying the shared df
                df_co2 = df_co2.copy()
                today_start = _now_dk().normalize()
                df_co2 = df_co2[df_co2["Time"] >= today_start]
                
            if not df_co2.empty:
                fig_co2 = px.line(
                    df_co2, x="Time", y="gCO2_per_kWh",
                    title="CO₂ Emission (g/kWh)",
                    labels={"Time": "Time", "gCO2_per_kWh": "Emission (g/kWh)"},
                    color_discrete_sequence=["#2ca02c"] # Green for CO2
                )
                
                now = _now_dk().round("1min")
                fig_co2.add_vline(x=now, line_width=2, line_dash="dash", line_color="red", opacity=0.7)
                fig_co2.add_annotation(x=now, y=df_co2["gCO2_per_kWh"].max(), text="Now", showarrow=False, yshift=10)
                
                fig_co2.update_layout(
                    hovermode="x unified", xaxis_rangeslider_visible=False,
                    yaxis_title="gCO₂/kWh", showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    uirevision="chart_state"
                )
                co2_xmin, co2_xmax = df_co2["Time"].min(), df_co2["Time"].max()
                _add_day_bands(fig_co2, co2_xmin, co2_xmax)
                fig_co2.update_xaxes(
                    showgrid=True, griddash="dot",
                    dtick=12 * 3600 * 1000,
                    tick0=_now_dk().normalize(),
                    tickformat="%H:%M\n%a %b %d",
                    range=[co2_xmin, co2_xmax],
                )
                fig_co2.update_yaxes(showgrid=True, griddash="dot")
                st.plotly_chart(fig_co2, width='stretch', height=250, key="chart_co2_forecast")
            else:
                st.warning("⚠️ Data source unavailable — energidataservice.dk could not be reached. CO₂ data will appear once the connection is restored.")
                
        except Exception as e:
            import requests
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                st.warning("⚠️ Data source unavailable — energidataservice.dk could not be reached.")
            else:
                st.error(f"Error loading CO2 data: {e}")

def render_gas_price():
    # --- 5. Natural Gas Balancing Price Block ---
    with st.container(border=True):
        st.markdown("#### 🔥 Natural Gas Price")
        
        try:
            # Fetch last 30 days
            df_gas = fetch_gas_prices(limit=100)
            
            if not df_gas.empty:
                # Filter to recent 35 days (roughly 1 month + some room)
                month_ago = _now_dk().normalize() - pd.Timedelta(days=35)
                df_plot = df_gas[df_gas["GasDay"] >= month_ago].copy()
                
                if not df_plot.empty:
                    fig_gas = go.Figure()
                    
                    fig_gas.add_trace(go.Scatter(
                        x=df_plot["GasDay"], y=df_plot["PurchasePriceDKK_kWh"],
                        mode='lines', name='Purchase Price',
                        line=dict(color='#d62728', width=2)
                    ))
                    
                    # Highlight Today
                    now = _now_dk().normalize()
                    fig_gas.add_vline(x=now, line_width=2, line_dash="dash", line_color="orange")
                    
                    fig_gas.update_layout(
                        title="Gas Balancing (Imbalance) Prices",
                        yaxis_title="DKK/kWh",
                        hovermode="x unified",
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=20, r=20, t=40, b=20),
                        uirevision="chart_state"
                    )
                    fig_gas.update_xaxes(showgrid=True, griddash="dot", tickformat="%b %d")
                    fig_gas.update_yaxes(showgrid=True, griddash="dot")
                    
                    st.plotly_chart(fig_gas, width='stretch', height=180, key="chart_gas_price")
                else:
                    st.warning("⚠️ Data source unavailable — energidataservice.dk could not be reached. Gas prices will appear once the connection is restored.")
            else:
                st.warning("⚠️ Data source unavailable — energidataservice.dk could not be reached. Gas prices will appear once the connection is restored.")
                
        except Exception as e:
            import requests
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                st.warning("⚠️ Data source unavailable — energidataservice.dk could not be reached.")
            else:
                st.error(f"Error loading gas price data: {e}")
