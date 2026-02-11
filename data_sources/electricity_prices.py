"""
Consolidated electricity price data module.
Handles fetching, processing, and combining actual and predicted electricity prices.
Includes ML prediction logic (formerly in ml_models/electricity_prediction.py).
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import certifi
import warnings
import json
from datetime import datetime, timedelta
from services.paths import results_dir

# ML Imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
except ImportError:
    pass # Handle missing deps gracefully if just fetching data

# -------- Configuration --------
EDS_PRICE_URL_OLD = "https://api.energidataservice.dk/dataset/Elspotprices"
EDS_PRICE_URL_NEW = "https://api.energidataservice.dk/dataset/DayAheadPrices"
TZ_DK = "Europe/Copenhagen"


# -------- Data Loading --------

def load_electricity_prices():
    """Loads electricity price prediction results."""
    csv_path = results_dir() / "Electricity_price_prediction_result.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df_price = pd.read_csv(csv_path)
    df_price["DateTime"] = pd.to_datetime(df_price["DateTime"], errors="coerce")
    df_price = df_price.dropna(subset=["DateTime"])
    return df_price


# -------- Data Fetching (EnergiDataService) --------

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dayahead_prices_latest(area: str = "DK1") -> pd.DataFrame:
    """Fetch latest day-ahead prices from EnergiDataService."""
    try:
        r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=200000", timeout=40, verify=certifi.where())
        r.raise_for_status()
    except requests.exceptions.SSLError:
        warnings.warn(f"SSL verification failed for {EDS_PRICE_URL_NEW}. Retrying without verification.")
        r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=200000", timeout=40, verify=False)
        r.raise_for_status()
    
    recs = r.json().get("records", [])
    if not recs: return pd.DataFrame()

    df = pd.DataFrame.from_records(recs)
    if "TimeDK" in df.columns: df = df.rename(columns={"TimeDK": "HourDK"})
    if "DayAheadPriceDKK" in df.columns: df = df.rename(columns={"DayAheadPriceDKK": "SpotPriceDKK"})
    if "DayAheadPriceEUR" in df.columns: df = df.rename(columns={"DayAheadPriceEUR": "SpotPriceEUR"})

    if "HourDK" not in df.columns or "PriceArea" not in df.columns: return pd.DataFrame()

    df = df[df["PriceArea"] == area].copy()
    if df.empty: return pd.DataFrame()

    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
    df = df.dropna(subset=["HourDK"]).sort_values("HourDK")
    df = df[~df["HourDK"].duplicated(keep="first")]

    if "SpotPriceDKK" in df.columns and df["SpotPriceDKK"].notna().any():
        df["price_dkk_per_kwh"] = df["SpotPriceDKK"].astype(float) / 1000.0
    elif "SpotPriceEUR" in df.columns and df["SpotPriceEUR"].notna().any():
        eur_to_dkk = 7.45
        df["price_dkk_per_kwh"] = df["SpotPriceEUR"].astype(float) * eur_to_dkk / 1000.0
    else:
        return pd.DataFrame()

    return df.set_index("HourDK")[["price_dkk_per_kwh"]]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_elspot_prices(area: str = "DK1") -> pd.DataFrame:
    """Fetch historical spot prices from EnergiDataService."""
    try:
        r = requests.get(f"{EDS_PRICE_URL_OLD}?limit=200000", timeout=40, verify=certifi.where())
        r.raise_for_status()
    except requests.exceptions.SSLError:
        warnings.warn(f"SSL verification failed for {EDS_PRICE_URL_OLD}. Retrying without verification.")
        r = requests.get(f"{EDS_PRICE_URL_OLD}?limit=200000", timeout=40, verify=False)
        r.raise_for_status()
    
    df = pd.DataFrame.from_records(r.json().get("records", []))
    if df.empty or "HourDK" not in df or "PriceArea" not in df or "SpotPriceDKK" not in df:
        return pd.DataFrame()
    
    df = df[df["PriceArea"] == area][["HourDK","SpotPriceDKK"]].copy()
    df["price_dkk_per_kwh"] = df["SpotPriceDKK"].astype(float) / 1000.0
    return (df.assign(HourDK=pd.to_datetime(df["HourDK"], errors="coerce"))
              .dropna(subset=["HourDK"])
              .sort_values("HourDK")
              .set_index("HourDK")[["price_dkk_per_kwh"]])


def load_unified_price_data(area: str = "DK1") -> pd.DataFrame:
    """Loads unified electricity price data (Actual + Predicted) for UI."""
    # from DT_dashboard.services.data_io import load_electricity_prices # Removed
    
    now = pd.Timestamp.now()
    today_start = now.normalize()
    
    try:
        df_actual = _fetch_dayahead_prices_latest(area)
        if df_actual.empty: df_actual = _fetch_elspot_prices(area)
    except:
        df_actual = pd.DataFrame()
    
    if not df_actual.empty:
        df_actual = df_actual[df_actual.index >= today_start].copy()
        df_actual = df_actual.reset_index()
        df_actual.columns = ["DateTime", "SpotPrice_DKK_per_kWh"]
        df_actual["Source"] = "Actual"
        latest_actual_date = df_actual["DateTime"].max()
    else:
        latest_actual_date = now
    
    try:
        df_pred = load_electricity_prices()
        df_pred = df_pred[["DateTime", "SpotPrice_DKK_per_kWh"]].copy()
        df_pred["Source"] = "Predicted"
        df_pred = df_pred[df_pred["DateTime"] > latest_actual_date]
    except:
        df_pred = pd.DataFrame()
    
    if not df_actual.empty and not df_pred.empty:
        df_combined = pd.concat([df_actual, df_pred], ignore_index=True)
    elif not df_actual.empty:
        df_combined = df_actual
    elif not df_pred.empty:
        df_combined = df_pred
    else:
        return pd.DataFrame(columns=["DateTime", "SpotPrice_DKK_per_kWh", "Source"])
    
    df_combined = df_combined.sort_values("DateTime").reset_index(drop=True)
    if not df_combined.empty:
        start_time = df_combined["DateTime"].min()
        end_time = df_combined["DateTime"].max()
        minute_range = pd.date_range(start=start_time, end=end_time, freq="1min")
        df_combined = df_combined.set_index("DateTime").reindex(minute_range, method="ffill")
        df_combined = df_combined.reset_index().rename(columns={"index": "DateTime"})
        return df_combined
    return df_combined


def fetch_electricity_prices_for_ml(area: str = "DK1") -> pd.DataFrame:
    """Fetch electricity prices for ML training (returns SpotPriceDKK)."""
    df_new = _fetch_dayahead_prices_latest(area)
    df_old = _fetch_elspot_prices(area)
    
    dfs = []
    if not df_new.empty:
        df_temp = df_new.reset_index()
        df_temp.columns = ["HourDK", "price_dkk_per_kwh"]
        dfs.append(df_temp)
    if not df_old.empty:
        df_temp = df_old.reset_index()
        df_temp.columns = ["HourDK", "price_dkk_per_kwh"]
        dfs.append(df_temp)
    
    if not dfs: return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df = (df.dropna(subset=["HourDK"])
            .sort_values("HourDK")
            .drop_duplicates(subset=["HourDK"], keep="first"))
    
    if pd.infer_freq(df["HourDK"].sort_values()) not in ("H", "h"):
        df = df.set_index("HourDK").resample("h").mean().reset_index()
    
    df = df.set_index("HourDK").sort_index()
    df["SpotPriceDKK"] = df["price_dkk_per_kwh"] * 1000.0
    return df[["SpotPriceDKK"]]


# -------- ML Prediction Logic --------

def _get_weather_unified(start_date, end_date):
    from data_sources.weather import fetch_weather_open_meteo
    # Coordinates for Aalborg
    LAT, LON = 57.048, 9.921
    df = fetch_weather_open_meteo(LAT, LON, start_date, end_date)
    if df.empty: return df
    
    # Rename for ML model compatibility
    # weather.py: temp, wind, prcp, wpgt, coco
    # ML expects: Temperature, WindSpeed, Precipitation, PeakGust, WeatherCondition
    return df.rename(columns={
        "temp": "Temperature",
        "wind": "WindSpeed",
        "prcp": "Precipitation",
        "wpgt": "PeakGust",
        "coco": "WeatherCondition"
    })


def _merge_ml_data(electricity_data, weather_data):
    merged = pd.merge(electricity_data, weather_data, left_index=True, right_index=True, how="inner")
    merged["hour"] = merged.index.hour
    merged["day_of_week"] = merged.index.dayofweek
    merged["is_weekend"] = merged["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    merged = merged.sort_index()
    feature_cols = ["WindSpeed", "Temperature", "Precipitation", "PeakGust", "WeatherCondition", "hour", "day_of_week", "is_weekend"]
    target_cols = ["SpotPriceDKK"]
    return merged, feature_cols, target_cols

def update_electricity_predictions():
    """Run full ML prediction pipeline and save results."""
    print("Starting Electricity Price Prediction...")
    
    # 1. Fetch Prices
    df_el = fetch_electricity_prices_for_ml()
    limit_date = pd.Timestamp("2025-09-30")
    df_el = df_el[df_el.index >= limit_date]
    if not df_el.empty: df_el = df_el.resample("h").mean()

    # 2. Fetch Weather
    # 2. Fetch Weather
    df_weather = _get_weather_unified(df_el.index.min(), df_el.index.max())

    # 3. Train
    merged, feats, targets = _merge_ml_data(df_el, df_weather)
    X, y = merged[feats], merged[targets]
    
    test_hours = 24
    X_train, X_test = X.iloc[:-test_hours], X.iloc[-test_hours:]
    y_train, y_test = y.iloc[:-test_hours], y.iloc[-test_hours:]
    
    # Clean NaNs
    bad_mask = ~np.isfinite(pd.to_numeric(y_train.squeeze(), errors='coerce'))
    X_train = X_train.drop(index=y_train.index[bad_mask])
    y_train = y_train.drop(index=y_train.index[bad_mask])

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, 
                         subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # 4. Predict Future (3 days)
    end_date = df_el.index.max()
    fut_start = end_date + timedelta(hours=1)
    fut_end = fut_start + timedelta(days=3) - timedelta(hours=1)
    
    df_fut_w = _get_weather_unified(fut_start, fut_end)
    df_fut_w["hour"] = df_fut_w.index.hour
    df_fut_w["day_of_week"] = df_fut_w.index.dayofweek
    df_fut_w["is_weekend"] = df_fut_w["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df_fut_w = df_fut_w.sort_index()

    preds = model.predict(df_fut_w)
    df_fut_pred = pd.DataFrame({"SpotPriceDKK": preds}, index=pd.date_range(fut_start, fut_end, freq="h"))

    # 5. Process Output & Save
    # Only keep future from 'today' perspective for the output file
    today = pd.Timestamp.now().normalize()
    full_pred = df_fut_pred[df_fut_pred.index >= today].copy()
    
    # Add Tariff logic
    def calc_tariff(row):
        h, m = row.name.hour, row.name.month
        is_summer = 4 <= m <= 9
        if 0 <= h < 6: return 8.67
        elif 17 <= h < 21: return 33.80 if is_summer else 78.01
        else: return 13.00 if is_summer else 26.00
    
    full_pred["SpotPrice_DKK_per_kWh"] = full_pred["SpotPriceDKK"] / 1000
    full_pred["Tariff_DKK"] = full_pred.apply(calc_tariff, axis=1) / 100
    full_pred["TotalPrice"] = full_pred["SpotPrice_DKK_per_kWh"] + full_pred["Tariff_DKK"]
    full_pred["Source"] = "Predicted"
    full_pred.index.name = "DateTime"

    df_out = full_pred.reset_index()
    
    # Save to CSV
    out_path = results_dir() / "Electricity_price_prediction_result.csv"
    cols = [c for c in ["DateTime", "SpotPrice_DKK_per_kWh", "TotalPrice", "Source"] if c in df_out.columns]
    df_out[cols].to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    # 6. Publish MQTT (Moved to mqtt_publisher.py)
    # Return result to caller
    return df_out
