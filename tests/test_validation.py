"""Unit tests for data validation, schema verification, and boundary filtering."""

import pytest
import pandas as pd
import numpy as np

from nyc_trip_duration.validation import (
    DataValidationError,
    validate_input_schema,
    check_missing_values,
    filter_nyc_bounds,
    filter_training_outliers,
)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Create a minimal valid DataFrame."""
    return pd.DataFrame(
        {
            "id": ["id001", "id002"],
            "vendor_id": [1, 2],
            "pickup_datetime": ["2016-03-01 12:00:00", "2016-03-01 18:00:00"],
            "pickup_longitude": [-73.985, -73.980],
            "pickup_latitude": [40.750, 40.755],
            "dropoff_longitude": [-73.975, -73.970],
            "dropoff_latitude": [40.760, 40.765],
            "passenger_count": [1, 2],
            "trip_duration": [600, 900],
        }
    )


def test_validate_input_schema_success(sample_valid_df):
    """Schema validation passes for complete DataFrame."""
    validate_input_schema(sample_valid_df, require_target=True)


def test_validate_input_schema_missing_column(sample_valid_df):
    """Schema validation raises DataValidationError when a column is missing."""
    df_missing = sample_valid_df.drop(columns=["vendor_id"])
    with pytest.raises(DataValidationError) as exc:
        validate_input_schema(df_missing)
    assert "vendor_id" in str(exc.value)


def test_validate_input_schema_empty():
    """Empty DataFrame raises DataValidationError."""
    with pytest.raises(DataValidationError):
        validate_input_schema(pd.DataFrame())


def test_check_missing_values():
    """Test detection of null entries."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": ["x", "y", "z"]})
    nulls = check_missing_values(df)
    assert nulls == {"a": 1}


def test_filter_nyc_bounds():
    """Test filtering coordinates outside NYC box."""
    df = pd.DataFrame(
        {
            "pickup_longitude": [-73.98, -100.0],  # -100 is outside NYC
            "pickup_latitude": [40.75, 40.75],
            "dropoff_longitude": [-73.97, -73.97],
            "dropoff_latitude": [40.76, 40.76],
        }
    )
    cleaned, audit = filter_nyc_bounds(df)
    assert len(cleaned) == 1
    assert audit["dropped_outside_nyc"] == 1


def test_filter_training_outliers():
    """Test filtering speed anomalies, zero distance, and duration cap."""
    df = pd.DataFrame(
        {
            "distance_km": [2.0, 0.0, 50.0, 100.0],
            "trip_duration": [
                600,     # valid: 10 min, 2 km -> 12 km/h
                600,     # 0 distance -> dropped
                60,      # 50 km in 1 min -> 3000 km/h -> dropped by speed
                10000,   # 100 km in 10000s = 36 km/h (speed valid), but > 120 minutes -> dropped by duration
            ],
        }
    )
    cleaned, audit = filter_training_outliers(df)
    assert len(cleaned) == 1
    assert audit["dropped_zero_distance"] == 1
    assert audit["dropped_implausible_speed"] == 1
    assert audit["dropped_excessive_duration"] == 1
