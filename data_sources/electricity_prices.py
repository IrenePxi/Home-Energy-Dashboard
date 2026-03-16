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

_TZ = "Europe/Copenhagen"
def _now_dk():
    """Current wall-clock time in Europe/Copenhagen, tz-naive."""
    return pd.Timestamp.now(tz=_TZ).replace(tzinfo=None)

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
        r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=200000", timeout=(5, 15), verify=certifi.where())
        r.raise_for_status()
    except requests.exceptions.SSLError:
        warnings.warn(f"SSL verification failed for {EDS_PRICE_URL_NEW}. Retrying without verification.")
        try:
            r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=200000", timeout=(5, 15), verify=False)
            r.raise_for_status()
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    
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
        r = requests.get(f"{EDS_PRICE_URL_OLD}?limit=200000", timeout=(5, 15), verify=certifi.where())
        r.raise_for_status()
    except requests.exceptions.SSLError:
        warnings.warn(f"SSL verification failed for {EDS_PRICE_URL_OLD}. Retrying without verification.")
        try:
            r = requests.get(f"{EDS_PRICE_URL_OLD}?limit=200000", timeout=(5, 15), verify=False)
            r.raise_for_status()
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    
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
    
    now = _now_dk()
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
        # Keep predictions for today AND future
        df_pred = df_pred[df_pred["DateTime"] >= today_start]
    except:
        df_pred = pd.DataFrame()
    
    if df_actual.empty and df_pred.empty:
        return pd.DataFrame(columns=["DateTime", "SpotPrice_DKK_per_kWh", "Source"])

    # Process reindexing separately for each source to handle overlapping time ranges
    parts = []
    for source_name, df_source in [("Actual", df_actual), ("Predicted", df_pred)]:
        if df_source.empty: continue
        
        # Sort and deduplicate
        df_s = df_source.sort_values("DateTime").drop_duplicates("DateTime").set_index("DateTime")
        
        # Reindex to 1-minute frequency
        start_time = df_s.index.min()
        end_time = df_s.index.max()
        minute_range = pd.date_range(start=start_time, end=end_time, freq="1min")
        
        df_s = df_s.reindex(minute_range, method="ffill")
        df_s = df_s.reset_index().rename(columns={"index": "DateTime"})
        df_s["Source"] = source_name
        parts.append(df_s)
    
    if not parts:
        return pd.DataFrame(columns=["DateTime", "SpotPrice_DKK_per_kWh", "Source"])
        
    df_combined = pd.concat(parts, ignore_index=True).sort_values(["DateTime", "Source"])
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
    """Run refined ML prediction pipeline: Validation -> Metrics -> Retrain -> Predict."""
    print("Starting Refined Electricity Price Prediction Pipeline...")
    
    # 1. Fetch Data
    df_el = fetch_electricity_prices_for_ml()
    limit_date = pd.Timestamp("2025-09-30")
    df_el = df_el[df_el.index >= limit_date]
    if not df_el.empty: df_el = df_el.resample("h").mean()

    df_weather = _get_weather_unified(df_el.index.min(), df_el.index.max())
    merged, feats, targets = _merge_ml_data(df_el, df_weather)
    X, y = merged[feats], merged[targets]

    # Clean NaNs
    bad_mask = ~np.isfinite(pd.to_numeric(y.squeeze(), errors='coerce'))
    X = X.drop(index=y.index[bad_mask])
    y = y.drop(index=y.index[bad_mask])

    # 2. Validation Split (Last 24h)
    test_hours = 24
    X_train, X_val = X.iloc[:-test_hours], X.iloc[-test_hours:]
    y_train, y_val = y.iloc[:-test_hours], y.iloc[-test_hours:]

    # 3. Train Model A (Validation Model)
    model_val = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, 
                             subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42)
    model_val.fit(X_train, y_train.values.ravel())

    # 4. Predict Held-out 24h and Calc Metrics
    val_preds = model_val.predict(X_val)
    y_val_arr = y_val.values.ravel()
    
    mae = float(np.mean(np.abs(y_val_arr - val_preds)))
    rmse = float(np.sqrt(np.mean((y_val_arr - val_preds)**2)))
    # MAPE handling zeros/near-zeros
    mape = float(np.mean(np.abs((y_val_arr - val_preds) / np.maximum(y_val_arr, 0.01))) * 100)

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2),
        "last_updated": _now_dk().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 5. Save Metrics
    metrics_path = results_dir() / "prediction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {metrics}")

    # 6. Retrain Model B on ALL data
    model_final = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, 
                               subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=42)
    model_final.fit(X, y.values.ravel())

    # 7. Predict Future (4 days: Today + 3 future)
    today_start = _now_dk().normalize()
    fut_start = today_start
    fut_end = fut_start + timedelta(days=4) - timedelta(hours=1)
    
    df_fut_w = _get_weather_unified(fut_start, fut_end)
    df_fut_w["hour"] = df_fut_w.index.hour
    df_fut_w["day_of_week"] = df_fut_w.index.dayofweek
    df_fut_w["is_weekend"] = df_fut_w["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df_fut_w = df_fut_w.sort_index()

    future_preds = model_final.predict(df_fut_w[feats])
    
    # 8. Combine Validation Predictions (for Today comparison) and Future predictions
    df_val_pred = pd.DataFrame({"SpotPriceDKK": val_preds}, index=y_val.index)
    df_fut_pred = pd.DataFrame({"SpotPriceDKK": future_preds}, index=pd.date_range(fut_start, fut_end, freq="h"))
    
    # Merge them, preferring fut_pred if overlap
    df_combined_pred = pd.concat([df_val_pred, df_fut_pred])
    df_combined_pred = df_combined_pred.sort_index()
    df_combined_pred = df_combined_pred[~df_combined_pred.index.duplicated(keep='last')]

    # Process Output
    def calc_tariff(row):
        h, m = row.name.hour, row.name.month
        is_summer = 4 <= m <= 9
        if 0 <= h < 6: return 8.67
        elif 17 <= h < 21: return 33.80 if is_summer else 78.01
        else: return 13.00 if is_summer else 26.00
    
    df_combined_pred["SpotPrice_DKK_per_kWh"] = df_combined_pred["SpotPriceDKK"] / 1000
    df_combined_pred["Tariff_DKK"] = df_combined_pred.apply(calc_tariff, axis=1) / 100
    df_combined_pred["TotalPrice"] = df_combined_pred["SpotPrice_DKK_per_kWh"] + df_combined_pred["Tariff_DKK"]
    df_combined_pred["Source"] = "Predicted"
    df_combined_pred.index.name = "DateTime"

    df_out = df_combined_pred.reset_index()
    
    # Save to CSV
    out_path = results_dir() / "Electricity_price_prediction_result.csv"
    cols = [c for c in ["DateTime", "SpotPrice_DKK_per_kWh", "TotalPrice", "Source"] if c in df_out.columns]
    df_out[cols].to_csv(out_path, index=False)
    print(f"Saved merged predictions to {out_path}")

    return df_out
