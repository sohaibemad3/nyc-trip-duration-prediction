# NYC Taxi Trip Duration - Data Documentation

This directory manages the datasets used for training, validating, and testing the trip duration prediction models.

## Directory Structure

```text
data/
├── README.md               # Data documentation and access guide
├── raw/                    # Original, uncleaned CSV files
│   └── .gitkeep
└── processed/              # Cleaned, feature-engineered intermediate files (if cached)
    └── .gitkeep
```

> **Note**: In accordance with open-source best practices and dataset terms, large CSV files and binary artifacts are excluded from version control via `.gitignore`.

## Dataset Overview

The dataset is derived from the **New York City Taxi & Limousine Commission (TLC) Yellow Taxi Trip Records**, published via NYC Open Data and hosted on Kaggle for the [NYC Taxi Trip Duration Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration).

The data covers yellow taxi trips in New York City from **January 1, 2016 to June 30, 2016**.

### Dataset Splits

In this repository, the full dataset (1,458,641 records) has been partitioned into disjoint training, validation, and test sets:

| Split | Filename | Record Count | Approximate Size | Target Included | Purpose |
|---|---|---|---|---|---|
| **Train** | `train.csv` | 1,000,000 | ~114 MB | Yes (`trip_duration`) | Model training and transformer fitting |
| **Validation** | `val.csv` | 229,319 | ~26 MB | Yes (`trip_duration`) | Hyperparameter tuning and model selection |
| **Test** | `test.csv` | 229,322 | ~26 MB | Yes (`trip_duration`) | Final holdout evaluation and benchmarking |

### Schema & Column Definitions

| Column | Data Type | Description |
|---|---|---|
| `id` | `object` | Unique alphanumeric identifier for each trip record |
| `vendor_id` | `int64` | Taxi technology provider code (1 = Creative Mobile Technologies; 2 = VeriFone Inc.) |
| `pickup_datetime` | `object` / `datetime` | Date and time when the meter was engaged (`YYYY-MM-DD HH:MM:SS`) |
| `passenger_count` | `int64` | Number of passengers in the vehicle (driver-entered) |
| `pickup_longitude` | `float64` | Longitude coordinate where the trip started |
| `pickup_latitude` | `float64` | Latitude coordinate where the trip started |
| `dropoff_longitude`| `float64` | Longitude coordinate where the trip ended |
| `dropoff_latitude` | `float64` | Latitude coordinate where the trip ended |
| `store_and_fwd_flag` | `object` | Flag indicating whether trip record was held in vehicle memory before transmission (`Y` = store and forward, `N` = not a store and forward) |
| `trip_duration` | `int64` | Target variable: Duration of the trip in seconds |

## Data Acquisition & Placement

If cloning this repository from scratch:

1. Obtain the Kaggle NYC Taxi Trip Duration competition dataset or download TLC records directly:
   - Kaggle: `kaggle competitions download -c nyc-taxi-trip-duration`
   - NYC TLC: [NYC Open Data TLC Trip Record Data](https://data.cityofnewyork.us/)
2. Place `train.csv`, `val.csv`, and `test.csv` in the project root or under `data/raw/`.
   - The pipeline scripts check both root directory (`./train.csv`) and `data/raw/` automatically.

## Data Cleaning & Quality Criteria

1. **Geographic Bounding Box (NYC Metro Area)**:
   - Longitude: `[-74.03, -73.77]`
   - Latitude: `[40.63, 40.85]`
   Trips with pickup or dropoff coordinates outside these limits are removed as GPS errors or long-distance intercity trips.
2. **Zero-Distance Filtering**:
   Trips with `distance_km == 0` (same pickup and dropoff coordinates) are removed from training.
3. **Speed Plausibility**:
   Implied speeds $\le 1\text{ km/h}$ (stationary / traffic jams / errors) or $\ge 160\text{ km/h}$ (~100 mph, unrealistic for NYC traffic) are removed during training cleaning.
4. **Duration Cap**:
   Trips exceeding 120 minutes (2 hours) are removed from training as abnormal outliers.
