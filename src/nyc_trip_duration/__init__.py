"""NYC Trip Duration Prediction Package.

Production-grade ML pipeline for taxi trip duration regression.
"""

from nyc_trip_duration.config import (
    FINAL_FEATURES,
    TARGET_COL,
    LOG_TARGET_COL,
    NYC_LON_BOUNDS,
    NYC_LAT_BOUNDS,
)
from nyc_trip_duration.data import load_dataset, save_dataset
from nyc_trip_duration.features import (
    haversine_distance,
    add_distance_feature,
    extract_datetime_features,
    engineer_all_features,
    compute_log_target,
    inverse_log_target,
)
from nyc_trip_duration.validation import (
    validate_input_schema,
    filter_nyc_bounds,
    filter_training_outliers,
)
from nyc_trip_duration.preprocessing import NYCTripPreprocessor
from nyc_trip_duration.models import (
    build_model,
    build_ridge,
    build_xgboost,
    build_mlp,
    save_model,
    load_model,
)
from nyc_trip_duration.evaluation import (
    calculate_regression_metrics,
    format_metrics_table,
    save_metrics,
)

__version__ = "0.1.0"
__all__ = [
    "FINAL_FEATURES",
    "TARGET_COL",
    "LOG_TARGET_COL",
    "NYC_LON_BOUNDS",
    "NYC_LAT_BOUNDS",
    "load_dataset",
    "save_dataset",
    "haversine_distance",
    "add_distance_feature",
    "extract_datetime_features",
    "engineer_all_features",
    "compute_log_target",
    "inverse_log_target",
    "validate_input_schema",
    "filter_nyc_bounds",
    "filter_training_outliers",
    "NYCTripPreprocessor",
    "build_model",
    "save_model",
    "load_model",
    "calculate_regression_metrics",
    "format_metrics_table",
    "save_metrics",
]
