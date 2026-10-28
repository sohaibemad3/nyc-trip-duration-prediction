#!/usr/bin/env python3
"""Evaluate a trained model artifact on validation or test datasets."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyc_trip_duration.config import (
    MODELS_DIR,
    RESULTS_DIR,
)
from nyc_trip_duration.data import load_dataset
from nyc_trip_duration.validation import (
    validate_input_schema,
    filter_nyc_bounds,
    filter_training_outliers,
)
from nyc_trip_duration.features import (
    add_distance_feature,
    extract_datetime_features,
    compute_log_target,
)
from nyc_trip_duration.preprocessing import NYCTripPreprocessor
from nyc_trip_duration.models import load_model
from nyc_trip_duration.evaluation import (
    calculate_regression_metrics,
    format_metrics_table,
    save_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved NYC Trip Duration model on holdout data.",
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
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate against.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save evaluation metrics JSON. Defaults to reports/results/eval_<split>_metrics.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"NYC TRIP DURATION - MODEL EVALUATION ({args.split.upper()})")
    print("=" * 70)

    # 1. Load artifacts
    print("\n[1/4] Loading model and preprocessor artifacts...")
    model = load_model(args.model_path)
    preprocessor = NYCTripPreprocessor.load(args.preprocessor_path)
    print(f"  Loaded model:        {args.model_path}")
    print(f"  Loaded preprocessor: {args.preprocessor_path}")

    # 2. Load dataset
    print(f"\n[2/4] Loading '{args.split}' dataset...")
    df = load_dataset(args.split)
    validate_input_schema(df, require_target=True)
    print(f"  Raw records loaded: {len(df):,}")

    # 3. Clean and engineer features
    print("\n[3/4] Running feature engineering...")
    df = add_distance_feature(df)
    df, _ = filter_nyc_bounds(df)
    if args.split in ["train", "val"]:
        df, _ = filter_training_outliers(df)
    df = extract_datetime_features(df)
    df["log_trip_duration"] = compute_log_target(df["trip_duration"])
    print(f"  Records evaluated:  {len(df):,}")

    # 4. Transform and predict
    print("\n[4/4] Generating predictions and calculating metrics...")
    X = preprocessor.transform(df)
    y_true = df["log_trip_duration"].to_numpy()

    y_pred = model.predict(X)
    metrics = calculate_regression_metrics(y_true, y_pred)

    model_label = Path(args.model_path).stem.replace("_", " ").title()
    print("\n" + format_metrics_table({f"{model_label} on {args.split.upper()}": metrics}))

    output_path = (
        Path(args.output_metrics)
        if args.output_metrics
        else RESULTS_DIR / f"eval_{args.split}_metrics.json"
    )
    payload = {
        "model_path": str(args.model_path),
        "split": args.split,
        "record_count": len(df),
        "metrics": metrics,
    }
    save_metrics(payload, output_path)
    print(f"\nEvaluation metrics saved to: {output_path}\n")


if __name__ == "__main__":
    main()
