import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from services.scripts import run_py
from data_sources.electricity_prices import load_electricity_prices, load_unified_price_data
from data_sources.pv import load_pv_predictions
from data_sources.gas import fetch_gas_prices

# Callbacks to lock autorefresh
def start_price_update():
    st.session_state["updating_price"] = True

def start_pv_update():
    st.session_state["updating_pv"] = True

def start_weather_update():
    # Clear cache to force fresh fetch
    st.cache_data.clear()
    st.session_state["updating_weather"] = True

def start_co2_update():
    # Clear cache to force fresh fetch for CO2
    from data_sources.co2 import fetch_co2_prog
    fetch_co2_prog.clear()
    st.session_state["updating_co2"] = True

def start_gas_update():
    st.session_state["updating_gas"] = True

def render_electricity_price():
    # --- 1. Electricity Price Prediction Block ---
    with st.container(border=True):
        st.markdown("#### 💡 Electricity Price Prediction")
        
        if st.button("🔄 Update Price", on_click=start_price_update, key="btn_update_price"):
            pass
        
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
                        st.error("Prediction failed.")
                except Exception as e:
                    st.error(f"Failed: {e}")
                finally:
                    st.session_state["updating_price"] = False
                    st.session_state["long_running_task"] = False
                    st.rerun()


        # Display Price Charts

        try:
            # Loaded from top-level import
            df_price = load_unified_price_data()
            
            # Filter to today and tomorrow
            today_start = pd.Timestamp.now().normalize()
            tomorrow_end = today_start + pd.Timedelta(days=2)
            df_hourly_view = df_price[
                (df_price["DateTime"] >= today_start) & 
                (df_price["DateTime"] < tomorrow_end) &
                (df_price["Source"] == "Actual")
            ].copy()
            
            if not df_hourly_view.empty:
                # Bar chart logic
                df_bar = df_hourly_view.set_index("DateTime")[["SpotPrice_DKK_per_kWh"]].resample("h").mean().reset_index()
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
                
                now = pd.Timestamp.now().round("1min")
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
            else:
                st.info("No hourly price data available for today.")

            # Line Chart

            if not df_price.empty:
                start_plot = pd.Timestamp.now() - pd.Timedelta(days=2)
                end_plot = pd.Timestamp.now() + pd.Timedelta(days=3)
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
                now = pd.Timestamp.now().round("1min")
                fig.add_shape(type="line", x0=now, y0=0, x1=now, y1=1, xref="x", yref="paper", line=dict(color="red", width=2, dash="dot"))
                fig.add_annotation(x=now, y=1.05, xref="x", yref="paper", text="Now", showarrow=False, font=dict(color="red"))

                fig.update_layout(
                    title="Price Forecast Trend",
                    yaxis_title="DKK/kWh",
                    hovermode="x unified",
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=40, b=20),
                    uirevision="chart_state"
                )
                fig.update_xaxes(
                    type="date", ticklabelmode="period", showgrid=True, griddash="dot",
                    tickformatstops=[
                        dict(dtickrange=[None, 1000*60*60*12], value="%H:%M\n%b %d"),
                        dict(dtickrange=[1000*60*60*12, 1000*60*60*24*3], value="%H:%M\n%b %d"),
                        dict(dtickrange=[1000*60*60*24*3, None], value="%b %d\n(%a)"),
                    ]
                )
                st.plotly_chart(fig, width='stretch', height=250, key="chart_price_trend")

        except Exception as e:
            st.error(f"Error loading price data: {e}")

def render_pv_forecast():
    # --- 2. PV Power Prediction Block ---
    with st.container(border=True):
        st.markdown("#### ☀️ PV Power Prediction")
        
        if st.button("🔄 Update PV", on_click=start_pv_update, key="btn_update_pv"):
            pass

        if st.session_state.get("updating_pv", False):
            st.session_state["long_running_task"] = True
            with st.spinner("Running PV prediction script..."):
                try:
                    from data_sources.pv import run_pv_prediction
                    from services.mqtt_publisher import publish_pv_forecast
                    
                    df_res = run_pv_prediction()
                    if not df_res.empty:
                         if publish_pv_forecast(df_res):
                             st.success("Done! Data published.")
                         else:
                             st.warning("Prediction done, but MQTT publish failed.")
                    else:
                        st.error("Prediction failed. Check logs.")
                except Exception as e:
                    st.error(f"Failed: {e}")
                finally:
                    st.session_state["updating_pv"] = False
                    st.session_state["long_running_task"] = False
                    st.rerun()

        try:
            df_pv = load_pv_predictions()
            fig_pv = px.line(df_pv, x="DateTime", y="Corrected_PV", title="Predicted PV Power", labels={"DateTime": "Time", "Corrected_PV": "PV Power (W)"})
            now = pd.Timestamp.now().round("1min")
            fig_pv.add_vline(x=now, line_width=2, line_dash="dash", line_color="red")
            
            fig_pv.update_xaxes(
                type="date", ticklabelmode="period", showgrid=True, griddash="dot",
                tickformatstops=[
                    dict(dtickrange=[None, 1000*60*60*12], value="%H:%M\n%b %d"),
                    dict(dtickrange=[1000*60*60*12, 1000*60*60*24*3], value="%H:%M\n%b %d"),
                    dict(dtickrange=[1000*60*60*24*3, None], value="%b %d"),
                ]
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
        
        if st.button("🔄 Update Weather", on_click=start_weather_update, key="btn_update_weather"):
            pass
        
        if st.session_state.get("updating_weather", False):
             st.session_state["updating_weather"] = False
             st.rerun()

        try:
            from data_sources.weather import fetch_weather_open_meteo
            LAT, LON = 57.048, 9.921
            now = pd.Timestamp.now()
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
                    uirevision=True
                )
                fig_temp.update_xaxes(showgrid=True, griddash="dot", tickformat="%H:%M\n%b %d")
                fig_temp.update_yaxes(showgrid=True, griddash="dot")
                st.plotly_chart(fig_temp, width='stretch', height=250, key="chart_weather_temp")
            else:
                st.info("Weather data unavailable.")
                
        except Exception as e:
            st.error(f"Error loading weather data: {e}")

def render_co2_forecast(df_co2=None):
    # --- 4. CO2 Emission Forecast Block ---
    with st.container(border=True):
        st.markdown("#### 🌍 CO₂ Forecast")
        
        if st.button("🔄 Update CO2", on_click=start_co2_update, key="btn_update_co2"):
            pass
        
        if st.session_state.get("updating_co2", False):
             st.session_state["updating_co2"] = False
             st.rerun()
             
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
                today_start = pd.Timestamp.now().normalize()
                df_co2 = df_co2[df_co2["Time"] >= today_start]
                
            if not df_co2.empty:
                fig_co2 = px.line(
                    df_co2, x="Time", y="gCO2_per_kWh",
                    title="CO₂ Emission (g/kWh)",
                    labels={"Time": "Time", "gCO2_per_kWh": "Emission (g/kWh)"},
                    color_discrete_sequence=["#2ca02c"] # Green for CO2
                )
                
                now = pd.Timestamp.now().round("1min")
                fig_co2.add_vline(x=now, line_width=2, line_dash="dash", line_color="red", opacity=0.7)
                fig_co2.add_annotation(x=now, y=df_co2["gCO2_per_kWh"].max(), text="Now", showarrow=False, yshift=10)
                
                fig_co2.update_layout(
                    hovermode="x unified", xaxis_rangeslider_visible=False,
                    yaxis_title="gCO₂/kWh", showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    uirevision="chart_state"
                )
                fig_co2.update_xaxes(showgrid=True, griddash="dot", tickformat="%H:%M\n%b %d")
                fig_co2.update_yaxes(showgrid=True, griddash="dot")
                st.plotly_chart(fig_co2, width='stretch', height=250, key="chart_co2_forecast")
            else:
                st.info("CO2 data unavailable.")
                
        except Exception as e:
            st.error(f"Error loading CO2 data: {e}")

def render_gas_price():
    # --- 5. Natural Gas Balancing Price Block ---
    with st.container(border=True):
        st.markdown("#### 🔥 Natural Gas Price")
        
        if st.button("🔄 Update Gas", on_click=start_gas_update, key="btn_update_gas"):
            pass
        
        if st.session_state.get("updating_gas", False):
            # Clear cache to force fresh fetch
            fetch_gas_prices.clear()
            st.session_state["updating_gas"] = False
            st.rerun()

        try:
            # Fetch last 30 days
            df_gas = fetch_gas_prices(limit=100)
            
            if not df_gas.empty:
                # Filter to recent 35 days (roughly 1 month + some room)
                month_ago = pd.Timestamp.now().normalize() - pd.Timedelta(days=35)
                df_plot = df_gas[df_gas["GasDay"] >= month_ago].copy()
                
                if not df_plot.empty:
                    fig_gas = go.Figure()
                    
                    # Purchase Price
                    fig_gas.add_trace(go.Scatter(
                        x=df_plot["GasDay"], y=df_plot["PurchasePriceDKK_kWh"],
                        mode='lines', name='Purchase Price',
                        line=dict(color='#d62728', width=2)
                    ))
                    
                    # Sales Price
                    fig_gas.add_trace(go.Scatter(
                        x=df_plot["GasDay"], y=df_plot["SalesPriceDKK_kWh"],
                        mode='lines', name='Sales Price',
                        line=dict(color='#1f77b4', width=2, dash='dot')
                    ))
                    
                    # Highlight Today
                    now = pd.Timestamp.now().normalize()
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
                    st.info("No recent gas price data found.")
            else:
                st.info("Gas price data unavailable.")
                
        except Exception as e:
            st.error(f"Error loading gas price data: {e}")
