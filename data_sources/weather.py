"""
Weather data fetching from Open-Meteo.
Moved from app.py lines 352-425 — NO LOGIC CHANGES.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
import certifi
import warnings

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_open_meteo(lat: float, lon: float, start_date, end_date,
                             tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    """
    Hourly weather (GHI/DNI/DHI + temp + wind) from Open-Meteo (archive+forecast).
    Columns: ['ghi','dni','dhi','temp','wind'] indexed by local time (tz-naive).
    """
    def _get(base_url, s_date, e_date):
        url = (
            f"{base_url}?latitude={lat}&longitude={lon}"
            f"&hourly=shortwave_radiation,direct_normal_irradiance,"
            f"diffuse_radiation,temperature_2m,wind_speed_10m,"
            f"precipitation,wind_gusts_10m,weathercode"
            f"&start_date={s_date.strftime('%Y-%m-%d')}"
            f"&end_date={e_date.strftime('%Y-%m-%d')}"
            f"&timezone={tz.replace('/', '%2F')}"
        )
        try:
            r = requests.get(url, timeout=40, verify=certifi.where())
            r.raise_for_status()
        except requests.exceptions.SSLError:
            warnings.warn(f"SSL verification failed for {base_url}. Retrying without verification.")
            r = requests.get(url, timeout=40, verify=False)
            r.raise_for_status()
        h = r.json().get("hourly", {})
        if not h: return pd.DataFrame()
        df = pd.DataFrame(h).rename(columns={
            "shortwave_radiation": "ghi",
            "direct_normal_irradiance": "dni",
            "diffuse_radiation": "dhi",
            "temperature_2m": "temp",
            "wind_speed_10m": "wind",
            "precipitation": "prcp",
            "wind_gusts_10m": "wpgt",
            "weathercode": "coco"
        })
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time").sort_index()

    today = pd.Timestamp.now(tz).normalize()
    s_dt  = pd.Timestamp(start_date, tz=tz)
    e_dt  = pd.Timestamp(end_date,   tz=tz)

    past = _get("https://archive-api.open-meteo.com/v1/archive", s_dt, min(e_dt, today - pd.Timedelta(hours=1))) if s_dt < today else pd.DataFrame()
    futr = _get("https://api.open-meteo.com/v1/forecast", max(s_dt, today), e_dt) if e_dt >= today else pd.DataFrame()

    parts = [p for p in (past, futr) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["ghi","dni","dhi","temp","wind","prcp","wpgt","coco"])
    return pd.concat(parts).sort_index()


def _clean_hourly_index(s: pd.Series) -> pd.Series:
    """Make hourly series safe for reindex: tz-naive, on-the-hour, unique, sorted."""
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)   # drop tz
    s.index = s.index.floor("h")              # snap to hour
    # if duplicates remain (DST, API dup rows), keep the first (or .mean() if you prefer)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s

def daily_temperature_with_note(idx_min: pd.DatetimeIndex, weather_hr: pd.DataFrame) -> tuple[pd.Series, str|None]:
    note = None
    if weather_hr is None or weather_hr.empty or "temp" not in weather_hr:
        hrs = (idx_min - idx_min[0]).total_seconds()/3600.0
        placeholder = 10.0 + 6.0*np.sin(2*np.pi*(hrs-15)/24.0)
        return pd.Series(placeholder, index=idx_min, name="Tout_C"), \
               "No temperature data available. Showing a smooth placeholder curve."

    s_h = _clean_hourly_index(weather_hr["temp"].astype(float))

    start_h = idx_min[0].replace(minute=0, second=0, microsecond=0)
    end_h   = idx_min[-1].floor("h")
    idx_h   = pd.date_range(start=start_h, end=end_h, freq="h")

    s_h_aligned = s_h.reindex(idx_h)
    miss_h = int(s_h_aligned.isna().sum())
    if miss_h > 0:
        s_h_aligned = s_h_aligned.interpolate(limit_direction="both").bfill().ffill()
        note = f"Filled {miss_h} missing temperature points by interpolation."

    s_min = s_h_aligned.reindex(idx_min).interpolate().bfill().ffill().astype(float)
    return s_min.rename("Tout_C"), note
