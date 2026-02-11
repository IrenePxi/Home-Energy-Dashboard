import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_gas_prices(limit: int = 100) -> pd.DataFrame:
    """
    Fetches Natural Gas Daily Balancing Prices from Energi Data Service.
    Dataset: GasDailyBalancingPrice
    """
    url = f"https://api.energidataservice.dk/dataset/GasDailyBalancingPrice?limit={limit}&sort=GasDay DESC"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        records = data.get("records", [])
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        # Convert GasDay to datetime
        df["GasDay"] = pd.to_datetime(df["GasDay"])
        # Sorting DESC in API means newest first, but for Plotly we likely want chronological
        df = df.sort_values("GasDay")
        
        return df
    except Exception as e:
        st.error(f"Error fetching gas prices: {e}")
        return pd.DataFrame()
