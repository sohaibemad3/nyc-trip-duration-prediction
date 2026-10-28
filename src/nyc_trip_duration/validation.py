"""Data validation and quality checks for NYC taxi trip records."""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from nyc_trip_duration.config import (
    RAW_REQUIRED_COLUMNS,
    TARGET_COL,
    NYC_LON_BOUNDS,
    NYC_LAT_BOUNDS,
    MIN_SPEED_KMH,
    MAX_SPEED_KMH,
    MAX_DURATION_MINUTES,
)


class DataValidationError(ValueError):
    """Raised when an input DataFrame fails required schema or integrity checks."""
    pass


def validate_input_schema(
    df: pd.DataFrame,
    require_target: bool = False,
    extra_cols: Optional[List[str]] = None,
) -> None:
    """Validate that the DataFrame contains required columns with non-empty rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check.
    require_target : bool, default=False
        Whether `trip_duration` must be present.
    extra_cols : Optional[List[str]], default=None
        Additional columns required.

    Raises
    ------
    DataValidationError
        If any required column is missing or if the DataFrame is empty.
    """
    if df is None or len(df) == 0:
        raise DataValidationError("Input DataFrame is empty or None.")

    required = list(RAW_REQUIRED_COLUMNS)
    if require_target:
        required.append(TARGET_COL)
    if extra_cols:
        required.extend(extra_cols)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Input DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Return dictionary of column names and their missing value counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    Dict[str, int]
        Mapping of column name to number of null entries.
    """
    null_counts = df.isnull().sum()
    return {col: int(count) for col, count in null_counts.items() if count > 0}


def filter_nyc_bounds(
    df: pd.DataFrame,
    lon_bounds: Tuple[float, float] = NYC_LON_BOUNDS,
    lat_bounds: Tuple[float, float] = NYC_LAT_BOUNDS,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter records whose pickup or dropoff coordinates lie within NYC bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    lon_bounds : Tuple[float, float]
        (min_longitude, max_longitude)
    lat_bounds : Tuple[float, float]
        (min_latitude, max_latitude)

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned DataFrame and audit report dictionary of row counts.
    """
    initial_count = len(df)
    mask = (
        df["pickup_longitude"].between(*lon_bounds)
        & df["pickup_latitude"].between(*lat_bounds)
        & df["dropoff_longitude"].between(*lon_bounds)
        & df["dropoff_latitude"].between(*lat_bounds)
    )
    cleaned = df[mask].copy()
    dropped = initial_count - len(cleaned)

    audit = {
        "initial_rows": initial_count,
        "retained_rows": len(cleaned),
        "dropped_outside_nyc": dropped,
    }
    return cleaned, audit


def filter_training_outliers(
    df: pd.DataFrame,
    min_distance_km: float = 0.0,
    min_speed_kmh: float = MIN_SPEED_KMH,
    max_speed_kmh: float = MAX_SPEED_KMH,
    max_duration_minutes: float = MAX_DURATION_MINUTES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter physical anomalies and extreme outliers from training/validation data.

    Applied only when `trip_duration` and `distance_km` are available.

    Checks:
    1. distance_km > min_distance_km (trips that actually moved)
    2. real_speed_kmh within [min_speed_kmh, max_speed_kmh]
    3. duration_minutes <= max_duration_minutes

    Parameters
    ----------
    df : pd.DataFrame
        Training or validation DataFrame.
    min_distance_km : float, default=0.0
        Minimum distance threshold.
    min_speed_kmh : float, default=1.0
        Minimum plausible average speed in km/h.
    max_speed_kmh : float, default=160.0
        Maximum plausible average speed in km/h (~100 mph).
    max_duration_minutes : float, default=120.0
        Maximum duration threshold in minutes.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and detailed audit report dictionary.
    """
    initial_rows = len(df)
    audit: Dict[str, Any] = {"initial_rows": initial_rows}

    filtered = df.copy()

    # 1. Zero distance check
    if "distance_km" in filtered.columns:
        dist_mask = filtered["distance_km"] > min_distance_km
        dropped_zero_dist = int((~dist_mask).sum())
        filtered = filtered[dist_mask]
        audit["dropped_zero_distance"] = dropped_zero_dist
    else:
        audit["dropped_zero_distance"] = 0

    # 2. Implied speed check (requires duration)
    if TARGET_COL in filtered.columns and "distance_km" in filtered.columns:
        dur_min = filtered[TARGET_COL] / 60.0
        speed = filtered["distance_km"] / (dur_min / 60.0)

        speed_mask = (speed >= min_speed_kmh) & (speed <= max_speed_kmh)
        dropped_speed = int((~speed_mask).sum())
        filtered = filtered[speed_mask]
        audit["dropped_implausible_speed"] = dropped_speed

        dur_mask = (filtered[TARGET_COL] / 60.0) <= max_duration_minutes
        dropped_dur = int((~dur_mask).sum())
        filtered = filtered[dur_mask]
        audit["dropped_excessive_duration"] = dropped_dur
    else:
        audit["dropped_implausible_speed"] = 0
        audit["dropped_excessive_duration"] = 0

    audit["final_retained_rows"] = len(filtered)
    audit["total_dropped"] = initial_rows - len(filtered)

    return filtered, audit
