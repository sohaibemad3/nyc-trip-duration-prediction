#!/usr/bin/env python3
"""Run batch or single-trip duration inference using trained model artifacts."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyc_trip_duration.config import (
    MODELS_DIR,
    RAW_REQUIRED_COLUMNS,
)
from nyc_trip_duration.validation import validate_input_schema
from nyc_trip_duration.features import (
    add_distance_feature,
    extract_datetime_features,
    inverse_log_target,
)
from nyc_trip_duration.preprocessing import NYCTripPreprocessor
from nyc_trip_duration.models import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NYC taxi trip duration for new trips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "ridge_model.joblib"),
        help="Path to trained model artifact (.joblib).",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default=str(MODELS_DIR / "preprocessor.joblib"),
        help="Path to fitted preprocessor artifact (.joblib).",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to input CSV containing trip records to predict.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Path to output CSV with predictions appended.",
    )
    # CLI arguments for single-trip inference
    parser.add_argument("--vendor-id", type=int, default=1, help="Taxi vendor ID (1 or 2).")
    parser.add_argument("--pickup-datetime", type=str, default="2016-03-15 18:30:00", help="Trip pickup timestamp.")
    parser.add_argument("--pickup-longitude", type=float, default=-73.982, help="Pickup longitude.")
    parser.add_argument("--pickup-latitude", type=float, default=40.751, help="Pickup latitude.")
    parser.add_argument("--dropoff-longitude", type=float, default=-73.965, help="Dropoff longitude.")
    parser.add_argument("--dropoff-latitude", type=float, default=40.768, help="Dropoff latitude.")

    return parser.parse_args()


def predict_dataframe(
    df: pd.DataFrame,
    model,
    preprocessor: NYCTripPreprocessor,
) -> pd.DataFrame:
    """Run preprocessing and inference on an input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with required columns.
    model : Estimator
        Loaded model.
    preprocessor : NYCTripPreprocessor
        Fitted preprocessor.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with appended predictions:
        - `predicted_log_duration`
        - `predicted_duration_sec`
        - `predicted_duration_min`
    """
    validate_input_schema(df, require_target=False)

    df_prep = add_distance_feature(df)
    df_prep = extract_datetime_features(df_prep)

    X = preprocessor.transform(df_prep)
    pred_log = model.predict(X)
    pred_log = np.clip(pred_log, 0.0, None)

    pred_sec = inverse_log_target(pred_log)
    pred_min = pred_sec / 60.0

    result = df.copy()
    result["distance_km"] = df_prep["distance_km"]
    result["predicted_log_duration"] = pred_log
    result["predicted_duration_sec"] = np.round(pred_sec, 1)
    result["predicted_duration_min"] = np.round(pred_min, 2)
    return result


def main():
    args = parse_args()
    print("=" * 70)
    print("NYC TRIP DURATION - INFERENCE ENGINE")
    print("=" * 70)

    # 1. Load artifacts
    model = load_model(args.model_path)
    preprocessor = NYCTripPreprocessor.load(args.preprocessor_path)

    # 2. Check input mode: CSV or CLI single trip
    if args.input_csv:
        input_path = Path(args.input_csv)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found at {input_path}")
        print(f"\nProcessing batch CSV: {input_path}")
        df = pd.read_csv(input_path)
    else:
        print("\nProcessing single trip sample from CLI arguments:")
        df = pd.DataFrame(
            [
                {
                    "vendor_id": args.vendor_id,
                    "pickup_datetime": args.pickup_datetime,
                    "pickup_longitude": args.pickup_longitude,
                    "pickup_latitude": args.pickup_latitude,
                    "dropoff_longitude": args.dropoff_longitude,
                    "dropoff_latitude": args.dropoff_latitude,
                }
            ]
        )

    # 3. Generate predictions
    predictions = predict_dataframe(df, model, preprocessor)

    # 4. Display or save output
    if args.input_csv:
        out_path = Path(args.output_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out_path, index=False)
        print(f"Batch predictions saved to: {out_path} ({len(predictions):,} records)")
    else:
        row = predictions.iloc[0]
        print("\n--- Prediction Result ---")
        print(f"Pickup:             {row['pickup_datetime']}")
        print(f"Route:              ({row['pickup_latitude']:.4f}, {row['pickup_longitude']:.4f}) -> ({row['dropoff_latitude']:.4f}, {row['dropoff_longitude']:.4f})")
        print(f"Estimated Distance: {row['distance_km']:.2f} km")
        print(f"Predicted Duration: {row['predicted_duration_min']:.1f} minutes ({int(row['predicted_duration_sec'])} seconds)")
        print(f"Log(1 + Duration):  {row['predicted_log_duration']:.4f}\n")


if __name__ == "__main__":
    main()
