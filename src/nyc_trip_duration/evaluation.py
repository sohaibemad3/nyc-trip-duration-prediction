"""Evaluation metrics and reporting for NYC Trip Duration regression models."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nyc_trip_duration.features import inverse_log_target


def calculate_regression_metrics(
    y_true_log: Union[pd.Series, np.ndarray],
    y_pred_log: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Calculate comprehensive regression metrics on both log and original scales.

    Primary historical metrics:
    - log_r2: R² score computed on log(1 + trip_duration).
    - log_rmse: Root Mean Squared Error on log(1 + trip_duration), equivalent
      to RMSLE on the raw trip_duration in seconds.

    Real-scale metrics (inverted back to seconds):
    - raw_mae_sec: Mean Absolute Error in seconds.
    - raw_rmse_sec: Root Mean Squared Error in seconds.
    - raw_mae_min: Mean Absolute Error in minutes.

    Parameters
    ----------
    y_true_log : Union[pd.Series, np.ndarray]
        Ground truth log-transformed trip duration.
    y_pred_log : Union[pd.Series, np.ndarray]
        Predicted log-transformed trip duration.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed evaluation metrics.
    """
    y_true_arr = np.asarray(y_true_log, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred_log, dtype=np.float64)

    # 1. Log-scale metrics (historical methodology)
    log_r2 = float(r2_score(y_true_arr, y_pred_arr))
    log_mse = mean_squared_error(y_true_arr, y_pred_arr)
    log_rmse = float(np.sqrt(log_mse))
    log_mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

    # 2. Inverted real-scale metrics (seconds and minutes)
    y_true_raw = inverse_log_target(y_true_arr)
    # Clip negative predictions to 0 before expm1
    y_pred_clipped = np.clip(y_pred_arr, 0.0, None)
    y_pred_raw = inverse_log_target(y_pred_clipped)

    raw_mae_sec = float(mean_absolute_error(y_true_raw, y_pred_raw))
    raw_rmse_sec = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    raw_mae_min = raw_mae_sec / 60.0
    raw_rmse_min = raw_rmse_sec / 60.0

    return {
        "r2": log_r2,
        "rmse": log_rmse,  # RMSLE on original scale
        "rmsle": log_rmse,
        "mae_log": log_mae,
        "raw_mae_sec": raw_mae_sec,
        "raw_rmse_sec": raw_rmse_sec,
        "raw_mae_min": raw_mae_min,
        "raw_rmse_min": raw_rmse_min,
    }


def format_metrics_table(
    metrics_by_model: Dict[str, Dict[str, float]],
    title: str = "Model Performance Summary",
) -> str:
    """Format a Markdown table comparing model results.

    Parameters
    ----------
    metrics_by_model : Dict[str, Dict[str, float]]
        Nested dictionary {model_name: {metric_name: value}}.
    title : str
        Table heading.

    Returns
    -------
    str
        Formatted Markdown table string.
    """
    lines = [
        f"### {title}",
        "",
        "| Model | Log R² | Log RMSE (RMSLE) | Raw MAE (min) | Raw RMSE (min) |",
        "|---|---|---|---|---|",
    ]
    for model_name, m in metrics_by_model.items():
        lines.append(
            f"| **{model_name}** | {m.get('r2', float('nan')):.4f} | "
            f"{m.get('rmse', float('nan')):.4f} | "
            f"{m.get('raw_mae_min', float('nan')):.2f} min | "
            f"{m.get('raw_rmse_min', float('nan')):.2f} min |"
        )
    return "\n".join(lines)


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Persist evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics dictionary.
    output_path : Union[str, Path]
        Path to output JSON file.

    Returns
    -------
    Path
        Resolved path of saved metrics file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
