"""Unit tests for regression evaluation metrics and reporting."""

import numpy as np
import pytest

from nyc_trip_duration.evaluation import (
    calculate_regression_metrics,
    format_metrics_table,
)


def test_calculate_regression_metrics_perfect():
    """Perfect predictions yield R2=1.0 and RMSE=0.0."""
    y_true = np.array([5.0, 6.0, 7.0])
    metrics = calculate_regression_metrics(y_true, y_true)

    assert np.isclose(metrics["r2"], 1.0)
    assert np.isclose(metrics["rmse"], 0.0)
    assert np.isclose(metrics["raw_mae_sec"], 0.0)
    assert np.isclose(metrics["raw_mae_min"], 0.0)


def test_calculate_regression_metrics_with_error():
    """Test metric computation with realistic errors."""
    y_true = np.array([5.0, 6.0, 7.0])
    y_pred = np.array([5.1, 5.9, 7.2])
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert 0.0 < metrics["r2"] < 1.0
    assert metrics["rmse"] > 0.0
    assert metrics["raw_mae_min"] > 0.0


def test_format_metrics_table():
    """Test Markdown formatting contains expected columns and model names."""
    metrics = {
        "Ridge": {"r2": 0.7022, "rmse": 0.4093, "raw_mae_min": 3.78, "raw_rmse_min": 5.86}
    }
    table = format_metrics_table(metrics)
    assert "| Model |" in table
    assert "| **Ridge** |" in table
    assert "0.7022" in table
