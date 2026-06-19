"""Auto-refresh stale ML prediction CSVs on load."""
import warnings
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import streamlit as st

_TZ = "Europe/Copenhagen"


def _now_dk() -> pd.Timestamp:
    """Current wall-clock time in Europe/Copenhagen, tz-naive."""
    return pd.Timestamp.now(tz=_TZ).replace(tzinfo=None)


def get_max_prediction_time(csv_path: Path, datetime_col: str = "DateTime") -> Optional[pd.Timestamp]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=[datetime_col])
    except (ValueError, pd.errors.EmptyDataError):
        return None
    if df.empty:
        return None
    times = pd.to_datetime(df[datetime_col], errors="coerce").dropna()
    if times.empty:
        return None
    return times.max()


def is_prediction_stale(csv_path: Path, datetime_col: str = "DateTime") -> bool:
    """Predictions are stale when the latest row is before the current hour (DK time)."""
    max_dt = get_max_prediction_time(csv_path, datetime_col)
    if max_dt is None:
        return True
    return max_dt < _now_dk().floor("h")


def _fail_key(session_key: str) -> str:
    return f"{session_key}_auto_refresh_failed"


def clear_auto_refresh_failure(session_key: str) -> None:
    st.session_state.pop(_fail_key(session_key), None)


def get_auto_refresh_error(session_key: str) -> Optional[str]:
    return st.session_state.get(_fail_key(session_key))


def ensure_fresh_prediction(
    csv_path: Path,
    refresh_fn: Callable[[], None],
    session_key: str,
    spinner_message: str,
    datetime_col: str = "DateTime",
) -> bool:
    """
    Regenerate predictions when the CSV is missing or stale.
    Returns True if a refresh was attempted, False if data was already fresh.
    """
    if not is_prediction_stale(csv_path, datetime_col):
        return False

    if st.session_state.get(_fail_key(session_key)):
        return False

    try:
        with st.spinner(spinner_message):
            refresh_fn()
        clear_auto_refresh_failure(session_key)
        return True
    except Exception as exc:
        st.session_state[_fail_key(session_key)] = str(exc)
        warnings.warn(f"Auto-refresh failed for {csv_path.name}: {exc}")
        return False
