"""Integration tests for the complete end-to-end ML pipeline."""

import numpy as np
import pandas as pd
import pytest

from nyc_trip_duration.validation import (
    validate_input_schema,
    filter_nyc_bounds,
    filter_training_outliers,
)
from nyc_trip_duration.features import (
    add_distance_feature,
    extract_datetime_features,
    compute_log_target,
    inverse_log_target,
)
from nyc_trip_duration.preprocessing import NYCTripPreprocessor
from nyc_trip_duration.models import build_ridge
from nyc_trip_duration.evaluation import calculate_regression_metrics


def test_full_pipeline_end_to_end():
    """Test full cycle: raw data -> clean -> features -> preprocess -> train -> predict -> evaluate."""
    # Synthetic dataset
    np.random.seed(42)
    n_samples = 100

    raw_df = pd.DataFrame(
        {
            "id": [f"id{i:03d}" for i in range(n_samples)],
            "vendor_id": np.random.choice([1, 2], size=n_samples),
            "pickup_datetime": pd.date_range("2016-01-01", periods=n_samples, freq="h").astype(str),
            "passenger_count": np.random.randint(1, 5, size=n_samples),
            "pickup_longitude": np.random.uniform(-74.01, -73.80, size=n_samples),
            "pickup_latitude": np.random.uniform(40.65, 40.83, size=n_samples),
            "dropoff_longitude": np.random.uniform(-74.01, -73.80, size=n_samples),
            "dropoff_latitude": np.random.uniform(40.65, 40.83, size=n_samples),
            "store_and_fwd_flag": ["N"] * n_samples,
            "trip_duration": np.random.uniform(300, 2400, size=n_samples),
        }
    )

    # 1. Validation
    validate_input_schema(raw_df, require_target=True)

    # 2. Features
    df = add_distance_feature(raw_df)
    df, _ = filter_nyc_bounds(df)
    df, _ = filter_training_outliers(df)
    df = extract_datetime_features(df)
    df["log_trip_duration"] = compute_log_target(df["trip_duration"])

    # 3. Preprocessing
    preprocessor = NYCTripPreprocessor()
    X = preprocessor.fit_transform(df)
    y = df["log_trip_duration"].to_numpy()

    assert X.shape[0] == len(df)
    assert X.shape[1] == len(preprocessor.feature_names)

    # 4. Training
    model = build_ridge(alpha=1.0)
    model.fit(X, y)

    # 5. Prediction
    preds_log = model.predict(X)
    assert len(preds_log) == len(y)

    preds_raw = inverse_log_target(preds_log)
    assert np.all(preds_raw > 0)

    # 6. Evaluation
    metrics = calculate_regression_metrics(y, preds_log)
    assert "r2" in metrics
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0
