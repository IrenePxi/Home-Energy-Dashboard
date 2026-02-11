"""
CO2 data fetching from EnergiDataService.
Moved from app.py lines 143-144, 276-348 — NO LOGIC CHANGES.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
import certifi
import warnings
from datetime import datetime, date, time, timedelta

# -------- EnergiDataService endpoints --------
EDS_CO2_HIST_URL  = "https://api.energidataservice.dk/dataset/CO2Emis"
EDS_CO2_PROG_URL  = "https://api.energidataservice.dk/dataset/CO2EmisProg"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_co2_for_day(period_start:date, period_end:date, area) -> pd.Series:
    """Return local-naive 5-min gCO2/kWh series for the given calendar day. May contain NaNs."""
    tz = "Europe/Copenhagen"
    start_naive = datetime.combine(period_start, time(0, 0))
    end_naive   = datetime.combine(period_end + timedelta(days=1), time(0, 0))
    idx5 = pd.date_range(start=start_naive, end=end_naive, freq="5min", inclusive="left")


    url = "https://api.energidataservice.dk/dataset/CO2EmisProg?limit=200000"
    try:
        r = requests.get(url, timeout=40, verify=certifi.where())
        r.raise_for_status()
    except requests.exceptions.SSLError:
        warnings.warn(f"SSL verification failed for {url}. Retrying without verification.")
        r = requests.get(url, timeout=40, verify=False)
        r.raise_for_status()
    recs = r.json().get("records", [])
    if not recs:
        return pd.Series(index=idx5, dtype=float, name="gCO2_per_kWh")

    df = pd.DataFrame.from_records(recs)
    need = {"Minutes5UTC", "PriceArea", "CO2Emission"}
    if not need.issubset(df.columns):
        return pd.Series(index=idx5, dtype=float, name="gCO2_per_kWh")

    df = df.loc[df["PriceArea"] == area].copy()
    df["Time"] = (
        pd.to_datetime(df["Minutes5UTC"], utc=True)
          .dt.tz_convert(tz)
          .dt.tz_localize(None)
    )

    s = (df.rename(columns={"CO2Emission":"gCO2_per_kWh"})
           .set_index("Time")["gCO2_per_kWh"]
           .sort_index())
    
    s = s.groupby(level=0).mean()


    # Keep only that day and align to exact 5-min grid
    s = s.loc[(s.index >= idx5[0]) & (s.index <= idx5[-1])].reindex(idx5)
    return s.rename("gCO2_per_kWh")

def daily_co2_with_note(idx_min: pd.DatetimeIndex, period_start:date, period_end:date, area) -> tuple[pd.Series, str|None]:
    """
    Return minute-level CO₂ (g/kWh) where each 5-min value is held constant
    through its 5-minute block. Also reports how many 5-min points were missing.
    """
    # 5-min local-naive CO₂ for the calendar day (may contain NaNs at 5-min stamps)
    s5 = fetch_co2_for_day(period_start, period_end, area).rename("gCO2_per_kWh")  # expected 288 rows

    # Build a 5-min grid that covers the minute range
    start5 = idx_min[0].floor("5min")
    end5   = idx_min[-1].ceil("5min")
    idx5   = pd.date_range(start=start5, end=end5, freq="5min", inclusive="left")


    # Align to the 5-min grid
    s5_aligned = s5.reindex(idx5)
    miss5 = int(s5_aligned.isna().sum())
    note = None

    if s5_aligned.isna().all():
        # No API data at all → synthesize (then step-hold)
        hrs = (idx_min - idx_min[0]).total_seconds() / 3600.0
        s_min = pd.Series(250.0 + 100.0*np.sin(2*np.pi*(hrs - 15.0)/24.0),
                          index=idx_min, name="gCO2_per_kWh")
        note = "No CO₂ data from EnergiDataService for this day. Showing a smooth placeholder curve."
        return s_min, note

    # Fill only the missing *5-min* stamps (no interpolation within blocks)
    if miss5 > 0:
        s5_aligned = s5_aligned.ffill().bfill()
        note = f"Filled {miss5} missing CO₂ points by forward/backward fill on the 5-min grid."

    # Upsample to minutes with step-hold (constant within each 5-min slot)
    s_min = s5_aligned.reindex(idx_min).ffill().astype(float)
    return s_min.rename("gCO2_per_kWh"), note


@st.cache_data(ttl=300, show_spinner=False)
def fetch_co2_prog(area: str = "DK1", horizon_hours: int = 48) -> pd.DataFrame:
    """
    Pull latest CO2EmisProg rows (5-min resolution).
    - Filters by PriceArea
    - Returns DataFrame with 'Time', 'PriceArea', 'gCO2_per_kWh'
    """
    # 12 points per hour (5-min resolution)
    need = int(horizon_hours) * 12
    limit = max(200, need * 3)  # fetch generous chunk to ensure we get enough data

    url = f"{EDS_CO2_PROG_URL}?limit={limit}"
    
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        warnings.warn(f"Failed to fetch CO2 prognosis: {e}")
        return pd.DataFrame()

    recs = r.json().get("records", [])
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(recs)

    # Validate columns
    required = ["Minutes5UTC", "PriceArea", "CO2Emission"]
    if not set(required).issubset(df.columns):
        return pd.DataFrame()

    # Filter area
    df = df[df["PriceArea"] == area].copy()
    if df.empty:
        return pd.DataFrame()

    # Parse Time
    # API returns UTC. Convert to Europe/Copenhagen and remove tz info for plotting consistency
    df["Time"] = (
        pd.to_datetime(df["Minutes5UTC"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Copenhagen")
        .dt.tz_localize(None)
    )

    # Rename and clean
    df = (
        df.rename(columns={"CO2Emission": "gCO2_per_kWh"})
          [["Time", "PriceArea", "gCO2_per_kWh"]]
          .dropna(subset=["Time"])
          .sort_values("Time")
    )
    
    # Keep only future data or requested horizon? usually prognosis includes recent past + future.
    # We'll just return what we got, maybe tail(need) if strictly enforcing horizon.
    # But usually better to return all valid sorted points.
    
    return df.reset_index(drop=True)
