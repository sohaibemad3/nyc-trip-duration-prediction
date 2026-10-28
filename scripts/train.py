#!/usr/bin/env python3
"""Train regression models for NYC Taxi Trip Duration prediction."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path if script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyc_trip_duration.config import (
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
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
from nyc_trip_duration.models import build_model, save_model
from nyc_trip_duration.evaluation import (
    calculate_regression_metrics,
    format_metrics_table,
    save_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NYC Taxi Trip Duration prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "xgboost", "mlp"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional sampling fraction of training data (e.g., 0.1 for rapid smoke tests).",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=None,
        help="Path to save trained model artifact. Defaults to models/<model>_model.joblib.",
    )
    parser.add_argument(
        "--preprocessor-output",
        type=str,
        default=None,
        help="Path to save fitted preprocessor artifact. Defaults to models/preprocessor.joblib.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also run final evaluation on the test set.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic training.",
    )
    return parser.parse_args()


def prepare_data(df, is_training: bool = True):
    """Run data validation, cleaning, distance, and feature extraction."""
    validate_input_schema(df, require_target=True)

    # 1. Distance feature
    df = add_distance_feature(df)

    # 2. Filter geographic boundaries
    df, geo_audit = filter_nyc_bounds(df)

    # 3. Filter training outliers (if training or validation)
    if is_training:
        df, out_audit = filter_training_outliers(df)

    # 4. Temporal features
    df = extract_datetime_features(df)

    # 5. Log target
    df["log_trip_duration"] = compute_log_target(df["trip_duration"])
    return df


def main():
    args = parse_args()
    print("=" * 70)
    print(f"NYC TRIP DURATION - TRAINING PIPELINE ({args.model.upper()})")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Load training and validation datasets
    print("\n[1/5] Loading datasets...")
    df_train = load_dataset("train", sample_fraction=args.sample_fraction, random_state=args.random_seed)
    df_val = load_dataset("val")
    print(f"  Loaded raw train: {df_train.shape[0]:,} rows")
    print(f"  Loaded raw val:   {df_val.shape[0]:,} rows")

    # Step 2: Validate and prepare features
    print("\n[2/5] Cleaning and feature engineering...")
    df_train = prepare_data(df_train, is_training=True)
    df_val = prepare_data(df_val, is_training=True)
    print(f"  Train after cleaning: {df_train.shape[0]:,} rows")
    print(f"  Val after cleaning:   {df_val.shape[0]:,} rows")

    # Step 3: Fit preprocessor
    print("\n[3/5] Fitting preprocessor on training data...")
    preprocessor = NYCTripPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    y_train = df_train["log_trip_duration"].to_numpy()

    X_val = preprocessor.transform(df_val)
    y_val = df_val["log_trip_duration"].to_numpy()

    # Step 4: Build and train model
    print(f"\n[4/5] Training {args.model} model...")
    train_start = time.time()
    model = build_model(args.model, random_state=args.random_seed)
    model.fit(X_train, y_train)
    train_dur = time.time() - train_start
    print(f"  Model training completed in {train_dur:.2f} seconds.")

    # Step 5: Evaluate on validation set
    print("\n[5/5] Evaluating model performance...")
    val_pred = model.predict(X_val)
    val_metrics = calculate_regression_metrics(y_val, val_pred)

    results = {
        "model": args.model,
        "sample_fraction": args.sample_fraction,
        "train_rows": len(df_train),
        "validation_metrics": val_metrics,
    }

    print("\n" + format_metrics_table({f"{args.model.upper()} (Validation)": val_metrics}))

    if args.evaluate_test:
        print("\nEvaluating on Test Set...")
        df_test = load_dataset("test")
        df_test = prepare_data(df_test, is_training=False)
        X_test = preprocessor.transform(df_test)
        y_test = df_test["log_trip_duration"].to_numpy()

        test_pred = model.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, test_pred)
        results["test_metrics"] = test_metrics
        print("\n" + format_metrics_table({f"{args.model.upper()} (Test Set)": test_metrics}))

    # Save artifacts
    model_path = (
        Path(args.model_output)
        if args.model_output
        else MODELS_DIR / f"{args.model}_model.joblib"
    )
    prep_path = (
        Path(args.preprocessor_output)
        if args.preprocessor_output
        else MODELS_DIR / "preprocessor.joblib"
    )
    metrics_path = RESULTS_DIR / f"{args.model}_train_results.json"

    save_model(model, model_path)
    preprocessor.save(prep_path)
    save_metrics(results, metrics_path)

    print(f"\nArtifacts successfully persisted:")
    print(f"  Model:        {model_path}")
    print(f"  Preprocessor: {prep_path}")
    print(f"  Metrics:      {metrics_path}")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
