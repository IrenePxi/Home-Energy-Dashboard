import os
import sys
import json
import time
import pandas as pd
import numpy as np
import requests
import warnings
import datetime as dt
from datetime import datetime, timedelta, date
import pytz
from pytz import timezone

# Science/Math
import matplotlib.pyplot as plt

# PV Libs
import pvlib
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# Astral
from astral import LocationInfo, Observer
from astral.sun import sun, elevation

# ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from meteostat import Point, Hourly, Daily

# MQTT
from DT_dashboard.services.paths import results_dir

def load_pv_predictions():
    """Loads PV prediction results."""
    csv_path = results_dir() / "pv_prediction_result.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df_pv = pd.read_csv(csv_path)
    df_pv["DateTime"] = pd.to_datetime(df_pv["DateTime"])
    return df_pv

# --- Configuration ---
LAT, LON = 57.048, 9.921
ALTITUDE = 10
TZ = "Europe/Copenhagen"


def resource_path(relative_path):
    """ Get absolute path to resource, relative to this file """
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# 📌 Function to Calculate Solar Elevation Angle
def calculate_solar_elevation(lat, lon, timestamps):
    observer = Observer(latitude=lat, longitude=lon)
    elevations = []
    
    for timestamp in timestamps:
        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            # Assuming timestamps are in local time (Europe/Copenhagen) if naive, 
            # or UTC? Original script enforced UTC conversion for astral.
            # Only convert if naive?
            # Astral expects timezone aware or assumes UTC?
            # The original code did: timestamp.replace(tzinfo=pytz.utc)
            # which forces UTC. Let's stick to that if it works.
             timestamp_utc = timestamp.replace(tzinfo=pytz.utc)
        else:
             timestamp_utc = timestamp.astimezone(pytz.utc)
             
        elevation_angle = elevation(observer, timestamp_utc)  
        elevations.append(elevation_angle)

    return elevations

def PVmodel_pvlib(start_date, end_date, weather_data):
    lat, lon = LAT, LON
    altitude = ALTITUDE
    tz_local = TZ

    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    location = Location(lat, lon, tz=tz_local, altitude=altitude)

    solar_position = location.get_solarposition(timestamps)
    # solar_elevation = solar_position["apparent_elevation"].clip(lower=0)

    # Compute Plane of Array (POA) Irradiance
    surface_tilt = 30
    surface_azimuth = 180
    
    # Ensure both DataFrames have unique indices
    weather_data = weather_data[~weather_data.index.duplicated(keep="first")]
    solar_position = solar_position[~solar_position.index.duplicated(keep="first")]

    # Align indices
    common_index = weather_data.index.intersection(solar_position.index)
    weather_data = weather_data.loc[common_index]
    solar_position = solar_position.loc[common_index]

    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=weather_data["dni"],
        ghi=weather_data["ghi"],
        dhi=weather_data["dhi"],
        solar_zenith=solar_position["zenith"],
        solar_azimuth=solar_position["azimuth"]
    )

    weather_inputs = pd.DataFrame({
        "temp_air": weather_data["temp"],
        "wind_speed": weather_data["wspd"]
    }, index=weather_data.index)

    weather_full = pd.concat([poa_irradiance, weather_inputs], axis=1)

    # Define PV Module and Inverter Parameters
    # Total system power (16 panels x 400W = 6400W)
    module_parameters = {'pdc0': 6400, 'gamma_pdc': -0.004}
    inverter_parameters = {"pdc0": 6400, "eta_inv_nom": 0.96}
    
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

    system = PVSystem(
        arrays=[Array(
            FixedMount(surface_tilt=surface_tilt, surface_azimuth=surface_azimuth),
            module_parameters=module_parameters,
            temperature_model_parameters=temperature_model_parameters
        )],
        inverter_parameters=inverter_parameters
    )

    mc = ModelChain(
        system, system.arrays[0], location=location, # Provide location if needed by some models
        dc_model='pvwatts', ac_model='pvwatts',
        aoi_model='physical', spectral_model='no_loss'
    )
    # ModelChain.run_model_from_poa() requires location to adhere to docs?
    # Actually ModelChain signature is (system, location, ...)
    # But I construct it above.
    # In recent pvlib versions, location is part of ModelChain init.
    
    # Let's fix ModelChain init to match original script:
    # mc = ModelChain(system, location, ...)
    mc = ModelChain(
        system,
        location,
        dc_model='pvwatts',  
        ac_model='pvwatts',
        aoi_model='physical',
        spectral_model='no_loss'
    )
    
    mc.run_model_from_poa(weather_full)
    pv_power_output = mc.results.ac * 0.001  # Convert W to kW

    pv_power_output = pv_power_output.dropna()
    pv_power_output = pv_power_output[~pv_power_output.index.isna()]
    pv_power_output.index = pd.to_datetime(pv_power_output.index)
    
    return pv_power_output.to_frame(name="PV_pvlib")

def add_solar_features(df, lat, lon):
    df = df[~df.index.isna()]

    def safe_sun_feature(x, key):
        try:
            return sun(Observer(lat, lon), date=x)[key].hour
        except:
            return np.nan

    # This can be slow for large indices, optimize if possible or keep as is
    # Using apply/map is okay for hourly data
    df["solar_noon"] = df.index.map(lambda x: safe_sun_feature(x, "noon"))
    df["sunrise"] = df.index.map(lambda x: safe_sun_feature(x, "sunrise"))
    df["sunset"] = df.index.map(lambda x: safe_sun_feature(x, "sunset"))
    df["sun_up"] = (df["solar_elevation"] > 0).astype(int)

    return df

def fetch_weather_data(location, start_date, end_date):
    from DT_dashboard.data_sources.weather import fetch_weather_open_meteo
    
    # Fetch unified data
    # weather.py returns: ghi, dni, dhi, temp, wind, prcp, wpgt, coco
    data = fetch_weather_open_meteo(location._lat, location._lon, start_date, end_date)
    
    if data.empty: return data

    # Rename to match PV module internal names
    data = data.rename(columns={"wind": "wspd"})
    
    # Calculate Solar Elevation (needed for features)
    timestamps = data.index
    data["solar_elevation"] = calculate_solar_elevation(location._lat, location._lon, timestamps)
    
    # Run PVLib Model (now uses actual GHI/DNI/DHI from API instead of estimation)
    pvlib_df = PVmodel_pvlib(start_date, end_date, data)
    data = data.merge(pvlib_df, left_index=True, right_index=True, how="inner")

    # Select and Rename for Output
    # PV module expects: Temperature, WindSpeed, Precipitation, WeatherCondition, PV_pvlib
    data = data.rename(columns={
        "temp": "Temperature",
        "wspd": "WindSpeed",
        "prcp": "Precipitation",
        "coco": "WeatherCondition"
    })
    
    # Keep necessary columns
    cols = ["Temperature", "WindSpeed", "Precipitation", "WeatherCondition", "solar_elevation", "PV_pvlib"]
    # If GHI is needed downstream? estimated_ghi was used...
    # The return only kept specific columns in original code: 
    # "temp", "wspd", "prcp", "coco", "solar_elevation", "estimated_ghi", "PV_pvlib" -> Renamed.
    # Estimated GHI was kept as "estimated_ghi" then (in original line 250 it was selected)
    # Wait, original line 250: "estimated_ghi".
    # But line 243: data["ghi"] = data["estimated_ghi"].
    # Is "estimated_ghi" used downstream?
    # `add_solar_features` uses `solar_elevation`.
    # `train_pv_forecast_model` uses `featuresin`.
    # `featuresin` is `merged_data.columns.drop(["PV1", "Forecast"])`.
    # So ALL columns returned by `fetch_weather_data` become features!
    # So if I replace `estimated_ghi` with actual `ghi`, the model training might change.
    # But that's good.
    # I should include `ghi` in the output, maybe named `GHI` or `estimated_ghi` (for compatibility? no, let's call it GHI).
    # But if I change feature names, does it matter?
    # `train_pv_forecast_model` drops "PV1" and "Forecast".
    # All other columns are features.
    # So I should return "GHI" instead of "estimated_ghi".
    # And check if `PV_pvlib` logic used "estimated_ghi"? No, it used "dni", "ghi", "dhi".
    
    # So I will include "ghi" (actual) in the output.
    data["GHI"] = data["ghi"] 
    # Add to cols
    cols.append("GHI")
    
    data = data[cols]

    data = add_solar_features(data, location._lat, location._lon)
    return data

def merge_pv_weather(pv_data, weather_data):
    merged_data = pv_data.merge(weather_data, left_index=True, right_index=True, how="inner")
    merged_data["hour"] = merged_data.index.hour
    merged_data["month"] = merged_data.index.month
    return merged_data

def train_pv_forecast_model(merged_data):
    featuresin = merged_data.columns.drop(["PV1", "Forecast"])
    featuresout = ["PV1"]

    merged_data = merged_data.dropna()
    X = merged_data[featuresin]
    y = merged_data[featuresout]

    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        objective='reg:squarederror', random_state=42
    )
    model.fit(X_train, y_train.values.ravel())

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"[OK] PV Model trained. Test MAE: {mae:.2f} kW")

    return model, X_test.index, y_test, predictions, featuresin, featuresout

def predict_pv_next_days(model, weather_forecast, featuresin, featuresout):
    predictions = model.predict(weather_forecast[featuresin])
    predicted_df = pd.DataFrame({
        "Date": weather_forecast.index,
        "Predicted_PV": predictions
    })
    return predicted_df

def convert_power(value):
    s_val = str(value)
    if " W" in s_val:
        return float(s_val.replace(" W", "")) / 1000
    elif "kW" in s_val:
        return float(s_val.replace(" kW", ""))
    elif "mW" in s_val:
        return float(s_val.replace(" mW", "")) / 1000000
    else:
        try:
            return float(value)
        except:
             return 0.0

def run_pv_prediction():
    """
    Main entry point for PV prediction.
    """
    print("Starting PV Prediction...")
    
    # Paths to CSV data (in pv_data subfolder)
    pv_gen_path = resource_path(os.path.join("pv_data", "PV generation hourly.csv"))
    pv_forc_path = resource_path(os.path.join("pv_data", "PV forecast hourly.csv"))

    if not os.path.exists(pv_gen_path):
        print(f"Error: {pv_gen_path} not found")
        return False
    
    pv_generation = pd.read_csv(pv_gen_path, parse_dates=["Time"], index_col="Time")
    pv_forecast = pd.read_csv(pv_forc_path, parse_dates=["Time"], index_col="Time")

    pv_data = pv_generation.merge(pv_forecast, left_index=True, right_index=True, suffixes=("_actual", "_forecast"))
    pv_data = pv_data[~pv_data.index.isna()].dropna()
    
    pv_data["PV1"] = pv_data["PV1"].apply(convert_power)
    pv_data["Forecast"] = pv_data["Forecast"].apply(convert_power)
    pv_data.loc[pv_data["PV1"] < 0.2, "PV1"] = 0

    # Get historical weather data
    location = Point(LAT, LON)
    start_date = pv_data.index.min()
    end_date = pv_data.index.max()
    weather_data = fetch_weather_data(location, start_date, end_date)

    if weather_data.empty:
        print("Error: Could not fetch validation weather data.")
        return False

    merged_data = merge_pv_weather(pv_data, weather_data).sort_index()

    # Train
    model, _, _, _, featurein, featureout = train_pv_forecast_model(merged_data)

    # Forecast next 3 days
    todaydate = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_start = todaydate
    forecast_end = forecast_start + timedelta(days=3) - timedelta(hours=1)
    
    weather_forecast = fetch_weather_data(location, forecast_start, forecast_end)
    if weather_forecast.empty:
        print("Error: Could not fetch forecast weather data.")
        return False
        
    weather_forecast["hour"] = weather_forecast.index.hour
    weather_forecast["month"] = weather_forecast.index.month
    weather_forecast = weather_forecast.sort_index(ascending=True)

    pv_forecast_result = predict_pv_next_days(model, weather_forecast, featurein, featureout)
    
    # Save & Publish
    pv_forecast_result = pv_forecast_result.reset_index(drop=True) # It has columns Date, Predicted_PV
    
    # Prepare payload
    pv_forecast_result.rename(columns={'Date': 'DateTime', 'Predicted_PV': 'Corrected_PV'}, inplace=True)
    pv_forecast_result['DateTime'] = pv_forecast_result['DateTime'].astype(str)
    
    # MQTT Publish (Moved to mqtt_publisher.py)
    # Return result to caller
    return pv_forecast_result

    # Save to CSV in results
    # Use relative path from project root if possible, or common results dir logic
    # In run_py case, cwd was project root.
    # Here, we assume we want to write to "results/" in CWD (project root).
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "pv_prediction_result.csv")
    
    # Ensure columns
    pv_forecast_result[["DateTime", "Corrected_PV"]].to_csv(output_path, index=False)
    print(f"[OK] Results saved to {output_path}")
    
    return pv_forecast_result
