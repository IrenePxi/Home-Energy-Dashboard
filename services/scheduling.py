import pandas as pd
import numpy as np
from datetime import time as _time, datetime, timedelta

def find_best_interval(
    df_price: pd.DataFrame, 
    df_co2: pd.DataFrame, 
    duration_hours: float, 
    w_cost: float = 0.5,
    earliest_time: _time = None,
    latest_time: _time = None
) -> dict | None:
    """
    Finds the best time interval to run a device based on weighted Cost and CO2 scores.
    
    Args:
        df_price: DataFrame with 'DateTime' and 'SpotPrice_DKK_per_kWh'.
        df_co2: DataFrame with 'Time' and 'gCO2_per_kWh'.
        duration_hours: Duration of the task in hours.
        w_cost: Weight for cost (0.0 to 1.0). 1.0 = Minimize Cost, 0.0 = Minimize CO2.
        earliest_time: Earliest allowed start time (e.g. 08:00).
        latest_time: Latest allowed end time (e.g. 22:00). If None, no restriction.
                     If end < start, it assumes overnight window.

    Returns:
        dict: {'start': datetime, 'end': datetime, 'avg_price': float, 'avg_co2': float, 'score': float} or None
    """
    
    # 1. Align Data
    # Resample to hourly to match price granularity (since CO2 is 5-min but price is hourly)
    # or interpolate price to 5-min? Hourly is sufficient for "Cheapest Hour" hints.
    
    if df_price.empty or df_co2.empty:
        return None

    # Merge on nearest hour
    df_p = df_price.copy()
    df_p["DateTime"] = pd.to_datetime(df_p["DateTime"])
    df_p = df_p.set_index("DateTime").sort_index()
    
    df_c = df_co2.copy()
    df_c["Time"] = pd.to_datetime(df_c["Time"])
    df_c = df_c.set_index("Time").sort_index()
    
    # Resample CO2 to hourly mean to match Price
    # Ensure we only use numeric columns to avoid TypeError
    if "gCO2_per_kWh" in df_c.columns:
        df_c_hourly = df_c[["gCO2_per_kWh"]].resample("h").mean()
    else:
        df_c_hourly = df_c.resample("h").mean(numeric_only=True)
    
    # Join (inner join to have both data)
    df = df_p[["SpotPrice_DKK_per_kWh"]].join(df_c_hourly[["gCO2_per_kWh"]], how="inner")
    
    if df.empty:
        return None

    # Filter for future only (from now)
    now = pd.Timestamp.now().floor("h")
    df = df[df.index >= now]
    
    # Limit lookahead (e.g. 48 hours)
    df = df.iloc[:48] 

    if df.empty:
        return None

    # 2. Filter by Time Window
    if earliest_time and latest_time:
        # This is tricky for a continuous multi-day index.
        # We want to keep hours that are "valid start times".
        # Valid start time t: [t, t+duration] must be within allowed window?
        # Or just start time? Usually start time.
        
        # Create mask
        # allowed_mask = df.index.map(lambda t: is_time_in_window(t.time(), earliest_time, latest_time))
        # But simple version: just filter rows where hour is in range? 
        # Window usually implies "Start between X and Y".
        
        e_hour = earliest_time.hour + earliest_time.minute/60
        l_hour = latest_time.hour + latest_time.minute/60
        
        # If l_hour < e_hour, it wraps overnight.
        # But for "Start Time", typically user says "Start between 8:00 and 20:00".
        
        valid_indices = []
        for t in df.index:
            t_hour = t.hour + t.minute/60
            is_in = False
            if l_hour >= e_hour:
                is_in = e_hour <= t_hour <= l_hour
            else:
                is_in = t_hour >= e_hour or t_hour <= l_hour
            
            if is_in:
                valid_indices.append(t)
                
        df = df.loc[valid_indices]
        
    if df.empty:
        return None

    # 3. Normalize
    p_vals = df["SpotPrice_DKK_per_kWh"].values
    c_vals = df["gCO2_per_kWh"].values
    
    p_min, p_max = np.min(p_vals), np.max(p_vals)
    c_min, c_max = np.min(c_vals), np.max(c_vals)
    
    norm_p = (p_vals - p_min) / (p_max - p_min) if p_max > p_min else 0.5
    norm_c = (c_vals - c_min) / (c_max - c_min) if c_max > c_min else 0.5
    
    # 4. Score
    df["score"] = w_cost * norm_p + (1 - w_cost) * norm_c
    
    # 5. Rolling Window Search
    # We need to find `duration_hours` consecutive blocks.
    # Since we filtered by time window, the index might not be consecutive!
    # e.g. 10:00, 11:00, ... 20:00, 08:00 (next day).
    # We cannot use simple rolling().
    # We must check each valid start time t, and see if [t, t+duration] exists and is contiguous.
    
    best_iv = None
    min_score = float("inf")
    
    # We need the original UNFILTERED dataframe to look up the full duration
    # Re-fetch full df for lookup
    df_full = df_p[["SpotPrice_DKK_per_kWh"]].join(df_c_hourly[["gCO2_per_kWh"]], how="inner")
    df_full = df_full[df_full.index >= now]
    
    duration_steps = int(np.ceil(duration_hours)) # Hourly steps
    
    for start_time in df.index:
        # Check if we have data for the full duration
        end_time = start_time + pd.Timedelta(hours=duration_hours)
        # We need integer hours coverage.
        
        # Get slice from full df
        # Note: selecting by time range includes endpoints? 
        # We want [start, start + duration).
        # Actually simplest is iloc if we find the position.
        
        try:
            loc_start = df_full.index.get_loc(start_time)
            # Check bounds
            if loc_start + duration_steps > len(df_full):
                continue
                
            # Compute score for this block
            # We can use the score from filtered df? No, the duration might span outside the "Start Window".
            # Usually strict window means "run entirely within"? Or "Start within"?
            # Let's assume "Entire run must be within window" if strict, or "Start within" if flexible.
            # User usually means "I want to start washing machine between X and Y".
            # The run can finish later.
            # Let's verify score using Normalized values computed on the *search space*?
            # Or global normalization? 
            # Better to calc score on the block using global norms.
            
            block_p = df_full["SpotPrice_DKK_per_kWh"].iloc[loc_start : loc_start + duration_steps]
            block_c = df_full["gCO2_per_kWh"].iloc[loc_start : loc_start + duration_steps]
            
            if len(block_p) < duration_steps: continue

            avg_p = block_p.mean()
            avg_c = block_c.mean()
            
            # Normalize avg against global range (approx score)
            n_p = (avg_p - p_min) / (p_max - p_min) if p_max > p_min else 0.5
            n_c = (avg_c - c_min) / (c_max - c_min) if c_max > c_min else 0.5
            
            score = w_cost * n_p + (1 - w_cost) * n_c
            
            if score < min_score:
                min_score = score
                best_iv = {
                    "start": start_time,
                    "end": start_time + pd.Timedelta(hours=duration_hours),
                    "avg_price": avg_p,
                    "avg_co2": avg_c,
                    "score": score
                }
                
        except KeyError:
            continue
            
    return best_iv
