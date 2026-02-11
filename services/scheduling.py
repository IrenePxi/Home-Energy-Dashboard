import pandas as pd
import numpy as np
from datetime import time as _time, datetime, timedelta

def find_best_interval(
    df_price: pd.DataFrame, 
    duration_hours: float, 
    earliest_time: _time = None,
    latest_time: _time = None,
    deadline_dt: datetime = None
) -> dict | None:
    """
    Finds the best time interval to run a device based on Cost.
    
    Args:
        df_price: DataFrame with 'DateTime' and 'SpotPrice_DKK_per_kWh'.
        duration_hours: Duration of the task in hours.
        earliest_time: Earliest allowed start time (e.g. 08:00).
        latest_time: Latest allowed end time (e.g. 22:00). If None, no restriction.
                     If end < start, it assumes overnight window.
        deadline_dt: Absolute deadline for the interval end. If provided, overrides 
                     the soft 'latest_time' check for the overall boundary.

    Returns:
        dict: {'start': datetime, 'end': datetime, 'avg_price': float, 'score': float} or None
    """
    
    if df_price.empty:
        return None

    # Merge on nearest hour
    df_p = df_price.copy()
    df_p["DateTime"] = pd.to_datetime(df_p["DateTime"])
    df_p = df_p.set_index("DateTime").sort_index()
    
    # Filter for future only (from now)
    now = pd.Timestamp.now().floor("h")
    
    # Prep for Search
    # We'll search across all future data
    df_full = df_p[df_p.index >= now].copy()
    if df_full.empty:
        return None

    # Global min/max for normalization (though with only cost, we just minimize the mean)
    p_vals = df_full["SpotPrice_DKK_per_kWh"].values
    p_min, p_max = np.min(p_vals), np.max(p_vals)

    # Search Loop
    best_iv = None
    min_score = float("inf")
    duration_steps = int(np.ceil(duration_hours))

    for start_time in df_full.index:
        try:
            loc_start = df_full.index.get_loc(start_time)
            if loc_start + duration_steps > len(df_full):
                break # No more data
            
            # Get slice for this interval
            block = df_full.iloc[loc_start : loc_start + duration_steps]
            
            # Absolute Deadline Check
            actual_end = start_time + pd.Timedelta(hours=duration_hours)
            if deadline_dt and actual_end > deadline_dt:
                break # Since df_full is sorted, all subsequent start times will also exceed deadline
            
            # Check Window Restraint (soft/periodic window)
            if earliest_time is not None and latest_time is not None:
                # Every step in the block must be inside the allowed window
                def is_in_window(dt):
                    t = dt.time()
                    if earliest_time <= latest_time:
                        return earliest_time <= t <= latest_time
                    else: # Overnight window (e.g., 20:00 to 06:00)
                        return t >= earliest_time or t <= latest_time
                
                # Check start and end of total duration (including fractional end)
                if not is_in_window(start_time) or not is_in_window(actual_end):
                    continue
                
                # Check intermediate points if duration is long
                if duration_hours > 1:
                    # Sample every hour to ensure the whole run fits
                    if not all(block.index.map(is_in_window)):
                        continue

            # Compute score for this block
            avg_p = block["SpotPrice_DKK_per_kWh"].mean()
            
            # Normalize against global range (p_min, p_max)
            # Since we only have cost, the normalized score is just the normalized price
            score = (avg_p - p_min) / (p_max - p_min) if p_max > p_min else 0.5
            
            if score < min_score:
                min_score = score
                best_iv = {
                    "start": start_time,
                    "end": start_time + pd.Timedelta(hours=duration_hours),
                    "avg_price": avg_p,
                    "score": score
                }
                   
        except Exception:
            continue
            
    return best_iv
