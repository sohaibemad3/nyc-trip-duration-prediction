"""Unit tests for feature engineering (distance, datetime, harmonics, log transform)."""

import numpy as np
import pandas as pd
import pytest

from nyc_trip_duration.features import (
    haversine_distance,
    add_distance_feature,
    extract_datetime_features,
    compute_log_target,
    inverse_log_target,
)


def test_haversine_distance_known_points():
    """Test Haversine calculation against known approximate coordinates.

    Empire State Building: (40.748817, -73.985428)
    Times Square: (40.758896, -73.985130)
    Approximate great-circle distance: ~1.12 km.
    """
    lat1, lon1 = 40.748817, -73.985428
    lat2, lon2 = 40.758896, -73.985130
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    assert 1.10 < dist < 1.15


def test_haversine_distance_zero():
    """Distance between identical coordinates is 0."""
    dist = haversine_distance(40.75, -73.98, 40.75, -73.98)
    assert np.isclose(dist, 0.0)


def test_add_distance_feature():
    """Test adding distance_km column to a DataFrame."""
    df = pd.DataFrame(
        {
            "pickup_latitude": [40.75],
            "pickup_longitude": [-73.98],
            "dropoff_latitude": [40.76],
            "dropoff_longitude": [-73.97],
        }
    )
    res = add_distance_feature(df)
    assert "distance_km" in res.columns
    assert res["distance_km"].iloc[0] > 0


def test_extract_datetime_features():
    """Test calendar, harmonic cyclical, and holiday extraction."""
    # New Year's Day 2016 at 8:00 AM (Friday - holiday & rush hour window)
    df = pd.DataFrame(
        {
            "pickup_datetime": ["2016-01-01 08:30:00", "2016-03-15 17:30:00"],
        }
    )
    res = extract_datetime_features(df)

    expected_cols = [
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_weekday",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "is_rush_hour",
        "is_late_night",
        "is_holiday",
    ]
    for col in expected_cols:
        assert col in res.columns

    # 2016-01-01 is an official US New York holiday
    assert res["is_holiday"].iloc[0] == 1

    # 2016-03-15 17:30 is a Tuesday evening rush hour
    assert res["is_rush_hour"].iloc[1] == 1


def test_log_target_invertibility():
    """Test that expm1(log1p(x)) reconstructs original positive values."""
    durations = np.array([1.0, 60.0, 720.0, 3600.0, 7200.0])
    log_dur = compute_log_target(durations)
    reconstructed = inverse_log_target(log_dur)
    np.testing.assert_allclose(reconstructed, durations, rtol=1e-6)
