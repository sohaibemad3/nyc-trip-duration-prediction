"""Feature engineering functions for distance, temporal, and cyclical features."""

from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd
import holidays

from nyc_trip_duration.config import (
    EARTH_RADIUS_KM,
    TARGET_COL,
    LOG_TARGET_COL,
)


def haversine_distance(
    lat1: Union[pd.Series, np.ndarray, float],
    lon1: Union[pd.Series, np.ndarray, float],
    lat2: Union[pd.Series, np.ndarray, float],
    lon2: Union[pd.Series, np.ndarray, float],
    radius_km: float = EARTH_RADIUS_KM,
) -> Union[pd.Series, np.ndarray]:
    """Calculate the great-circle distance between two geographic coordinates in kilometers.

    Uses the Haversine formula:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * arcsin(√a)
        d = R * c

    Parameters
    ----------
    lat1, lon1 : Coordinates of pickup points in decimal degrees.
    lat2, lon2 : Coordinates of dropoff points in decimal degrees.
    radius_km : float, default=6371.0
        Mean radius of the Earth in kilometers.

    Returns
    -------
    Union[pd.Series, np.ndarray]
        Great-circle distance in kilometers.
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    # Clip for floating point safety
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add great-circle `distance_km` column to DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain pickup/dropoff latitude and longitude columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added `distance_km` column.
    """
    df = df.copy()
    df["distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def extract_datetime_features(
    df: pd.DataFrame,
    datetime_col: str = "pickup_datetime",
) -> pd.DataFrame:
    """Extract temporal, calendar, cyclical, and holiday features from pickup datetime.

    Features generated:
    - pickup_month (1..12)
    - pickup_day (1..31)
    - pickup_hour (0..23)
    - pickup_weekday (0=Mon..6=Sun)
    - hour_sin, hour_cos (24-hour harmonic cycle)
    - weekday_sin, weekday_cos (7-day harmonic cycle)
    - is_weekend (binary flag, Sat/Sun)
    - is_rush_hour (binary flag, weekday 7-10am or 4-7pm)
    - is_late_night (binary flag, 11pm - 5am)
    - is_holiday (binary flag, official US New York state holidays)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    datetime_col : str, default='pickup_datetime'
        Name of timestamp column.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.
    """
    df = df.copy()
    dt_series = pd.to_datetime(df[datetime_col])

    df["pickup_month"] = dt_series.dt.month
    df["pickup_day"] = dt_series.dt.day
    df["pickup_hour"] = dt_series.dt.hour
    df["pickup_weekday"] = dt_series.dt.weekday

    # Cyclical representations
    df["hour_sin"] = np.sin(2.0 * np.pi * df["pickup_hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["pickup_hour"] / 24.0)
    df["weekday_sin"] = np.sin(2.0 * np.pi * df["pickup_weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2.0 * np.pi * df["pickup_weekday"] / 7.0)

    # Weekend flag
    df["is_weekend"] = (df["pickup_weekday"] >= 5).astype(int)

    # Rush hour flag (NYC peak traffic: weekday morning 7-10 AM or evening 4-7 PM)
    df["is_rush_hour"] = 0
    mask_weekday = df["pickup_weekday"] < 5
    mask_morning = (df["pickup_hour"] >= 7) & (df["pickup_hour"] < 10)
    mask_evening = (df["pickup_hour"] >= 16) & (df["pickup_hour"] < 19)
    df.loc[mask_weekday & (mask_morning | mask_evening), "is_rush_hour"] = 1

    # Late night flag (11 PM to 5 AM)
    df["is_late_night"] = (
        (df["pickup_hour"] >= 23) | (df["pickup_hour"] < 5)
    ).astype(int)

    # Official US/NY holidays
    unique_years = dt_series.dt.year.unique()
    ny_holidays = holidays.US(subdiv="NY", years=list(unique_years))
    trip_dates = dt_series.dt.date
    df["is_holiday"] = trip_dates.apply(lambda d: 1 if d in ny_holidays else 0)

    return df


def compute_log_target(y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Compute log1p transformed target: log(1 + trip_duration).

    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Trip duration in seconds.

    Returns
    -------
    Union[pd.Series, np.ndarray]
        Transformed log duration.
    """
    return np.log1p(y)


def inverse_log_target(y_log: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Invert log1p transformed predictions back to duration in seconds: exp(y) - 1.

    Parameters
    ----------
    y_log : Union[pd.Series, np.ndarray]
        Predicted log trip duration.

    Returns
    -------
    np.ndarray
        Trip duration in seconds.
    """
    return np.expm1(y_log)


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: distance + temporal features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or filtered trip record DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with distance and time features.
    """
    df = add_distance_feature(df)
    df = extract_datetime_features(df)
    if TARGET_COL in df.columns:
        df[LOG_TARGET_COL] = compute_log_target(df[TARGET_COL])
    return df
