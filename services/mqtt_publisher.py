import json
import pandas as pd
import paho.mqtt.publish as publish
import logging

# MQTT Config
MQTT_BROKER = "metr.dk"
MQTT_PORT = 1883
MQTT_AUTH = {"username": "pxi", "password": "eQzX7uefeTtnfBdrg1nIYTlAtZiR2K5o"}

# Topics
TOPIC_PRICE = "AAU/Model_pxi/Forecast/Electricityprice"
TOPIC_PV = "AAU/Model_pxi/Forecast/PV"

def _publish(topic, payload):
    try:
        publish.single(
            topic=topic,
            payload=json.dumps(payload),
            hostname=MQTT_BROKER,
            port=MQTT_PORT,
            qos=1,
            retain=True,
            auth=MQTT_AUTH
        )
        print(f"MQTT published to {topic}")
        return True
    except Exception as e:
        print(f"MQTT failed: {e}")
        return False

def publish_electricity_price(df: pd.DataFrame):
    """
    Publishes electricity price forecast.
    Expected columns: DateTime, SpotPrice_DKK_per_kWh, TotalPrice
    """
    if df.empty: return False
    
    # Ensure DateTime format
    df = df.copy()
    if not pd.api.types.is_string_dtype(df["DateTime"]):
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
    payload = df[["DateTime", "SpotPrice_DKK_per_kWh", "TotalPrice"]].to_dict(orient="records")
    return _publish(TOPIC_PRICE, payload)

def publish_pv_forecast(df: pd.DataFrame):
    """
    Publishes PV forecast.
    Expected columns: DateTime, Corrected_PV
    """
    if df.empty: return False

    # Ensure DateTime format
    df = df.copy()
    if not pd.api.types.is_string_dtype(df["DateTime"]):
         df["DateTime"] = pd.to_datetime(df["DateTime"]).astype(str)

    payload = df[["DateTime", "Corrected_PV"]].to_dict(orient="records")
    return _publish(TOPIC_PV, payload)
