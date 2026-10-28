"""Configuration parameters and paths for the NYC Trip Duration project."""

from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# Filesystem Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_DIR: Path = REPORTS_DIR / "results"

# Primary dataset filenames (checks data/raw first, falls back to project root)
TRAIN_FILENAME: str = "train.csv"
VAL_FILENAME: str = "val.csv"
TEST_FILENAME: str = "test.csv"

# ---------------------------------------------------------------------------
# Global Settings & Random Seed
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Geographic & Physical Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM: float = 6371.0

# NYC Metro Geographic Bounding Box (filters GPS drift and outside-city trips)
NYC_LON_BOUNDS: Tuple[float, float] = (-74.03, -73.77)
NYC_LAT_BOUNDS: Tuple[float, float] = (40.63, 40.85)

# Plausibility thresholds for training data cleaning
MIN_SPEED_KMH: float = 1.0
MAX_SPEED_KMH: float = 160.0
MAX_DURATION_MINUTES: float = 120.0

# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------
RAW_REQUIRED_COLUMNS: List[str] = [
    "vendor_id",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

TARGET_COL: str = "trip_duration"
LOG_TARGET_COL: str = "log_trip_duration"

NUMERICAL_COLS: List[str] = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "pickup_day",
    "pickup_month",
]

BINARY_COLS: List[str] = [
    "is_weekend",
    "is_rush_hour",
    "is_late_night",
    "is_holiday",
]

CATEGORICAL_COLS: List[str] = [
    "vendor_id",
]

# The canonical feature column order expected by all models
FINAL_FEATURES: List[str] = NUMERICAL_COLS + BINARY_COLS + ["vendor_id_2"]

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
RIDGE_CONFIG: Dict[str, Any] = {
    "alpha": 1.0,
    "random_state": RANDOM_SEED,
}

XGB_CONFIG: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

MLP_CONFIG: Dict[str, Any] = {
    "hidden_layer_sizes": (64, 32),
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}
