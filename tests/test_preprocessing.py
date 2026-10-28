"""Unit tests for NYCTripPreprocessor pipeline and persistence."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from nyc_trip_duration.config import FINAL_FEATURES
from nyc_trip_duration.preprocessing import NYCTripPreprocessor


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create a synthetic DataFrame with engineered features."""
    return pd.DataFrame(
        {
            "pickup_longitude": [-73.98, -73.97, -73.99],
            "pickup_latitude": [40.75, 40.74, 40.76],
            "dropoff_longitude": [-73.97, -73.96, -73.98],
            "dropoff_latitude": [40.76, 40.75, 40.77],
            "distance_km": [1.5, 3.2, 0.8],
            "hour_sin": [0.5, -0.5, 0.0],
            "hour_cos": [0.86, 0.86, 1.0],
            "weekday_sin": [0.78, 0.0, -0.78],
            "weekday_cos": [0.62, 1.0, 0.62],
            "pickup_day": [10, 15, 20],
            "pickup_month": [3, 4, 5],
            "is_weekend": [0, 1, 0],
            "is_rush_hour": [1, 0, 0],
            "is_late_night": [0, 0, 1],
            "is_holiday": [0, 0, 0],
            "vendor_id": [1, 2, 1],
        }
    )


def test_preprocessor_fit_transform(sample_feature_df):
    """Test fitting and transforming features produces expected matrix shape."""
    preprocessor = NYCTripPreprocessor()
    X = preprocessor.fit_transform(sample_feature_df)

    assert X.shape == (3, len(FINAL_FEATURES))
    assert np.all(X >= -1e-7) and np.all(X <= 1.0 + 1e-7)


def test_preprocessor_column_order(sample_feature_df):
    """Ensure preprocessor outputs columns in identical canonical order."""
    preprocessor = NYCTripPreprocessor()
    preprocessor.fit(sample_feature_df)
    assert preprocessor.feature_names == list(FINAL_FEATURES)


def test_preprocessor_save_and_load(tmp_path: Path, sample_feature_df):
    """Test serialization and loading of preprocessor."""
    prep = NYCTripPreprocessor()
    X_orig = prep.fit_transform(sample_feature_df)

    artifact_path = tmp_path / "prep.joblib"
    prep.save(artifact_path)
    assert artifact_path.is_file()

    loaded_prep = NYCTripPreprocessor.load(artifact_path)
    X_loaded = loaded_prep.transform(sample_feature_df)

    np.testing.assert_allclose(X_orig, X_loaded)
