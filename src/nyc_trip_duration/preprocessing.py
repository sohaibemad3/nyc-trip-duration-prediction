"""Preprocessing pipeline: Power transformation, categorical encoding, and MinMax scaling."""

from typing import List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import joblib

from nyc_trip_duration.config import (
    FINAL_FEATURES,
    NUMERICAL_COLS,
    BINARY_COLS,
    CATEGORICAL_COLS,
    MODELS_DIR,
)


class NYCTripPreprocessor(BaseEstimator, TransformerMixin):
    """Production-grade preprocessor preserving the exact notebook transformations.

    Pipeline:
    1. Distance Power Transformation (Yeo-Johnson, standardize=True) fitted on train.
    2. Categorical dummy encoding for `vendor_id` -> `vendor_id_2`.
    3. Alignment to canonical feature order (`FINAL_FEATURES`).
    4. MinMaxScaler normalization fitted on train and applied to inference.

    Parameters
    ----------
    standardize_distance : bool, default=True
        Whether to apply Yeo-Johnson standardization to distance_km.
    apply_scaling : bool, default=True
        Whether to apply MinMaxScaler to the final feature matrix.
    """

    def __init__(
        self,
        standardize_distance: bool = True,
        apply_scaling: bool = True,
    ):
        self.standardize_distance = standardize_distance
        self.apply_scaling = apply_scaling
        self.distance_transformer = PowerTransformer(
            method="yeo-johnson", standardize=True
        )
        self.scaler = MinMaxScaler()
        self.feature_names: List[str] = list(FINAL_FEATURES)
        self.is_fitted: bool = False

    def _prepare_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical dummies and guarantee all canonical features are present."""
        df = df.copy()

        # Handle vendor_id dummy encoding
        if "vendor_id" in df.columns:
            # If vendor_id == 2 -> 1, if 1 -> 0
            df["vendor_id_2"] = (df["vendor_id"] == 2).astype(int)
        elif "vendor_id_2" not in df.columns:
            # Default fallback if column absent
            df["vendor_id_2"] = 0

        # Fill any missing binary or numerical columns with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Return strictly the canonical columns in identical order
        return df[self.feature_names].copy()

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "NYCTripPreprocessor":
        """Fit Yeo-Johnson on distance_km and MinMaxScaler on the full feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature DataFrame (must have engineered features).
        y : None
            Ignored, present for Scikit-Learn API compatibility.

        Returns
        -------
        self
        """
        df = X.copy()

        # 1. Fit distance power transformer on training distances
        if self.standardize_distance and "distance_km" in df.columns:
            self.distance_transformer.fit(df[["distance_km"]])
            df["distance_km"] = self.distance_transformer.transform(df[["distance_km"]])

        # 2. Prepare columns and encode vendor_id
        df_prepared = self._prepare_columns(df)

        # 3. Fit MinMaxScaler
        if self.apply_scaling:
            self.scaler.fit(df_prepared)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        np.ndarray
            Scaled feature matrix ready for estimator consumption.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df = X.copy()

        # 1. Apply Yeo-Johnson transformation to distance_km
        if self.standardize_distance and "distance_km" in df.columns:
            df["distance_km"] = self.distance_transformer.transform(df[["distance_km"]])

        # 2. Prepare columns
        df_prepared = self._prepare_columns(df)

        # 3. Apply MinMaxScaler
        if self.apply_scaling:
            return self.scaler.transform(df_prepared)
        return df_prepared.to_numpy()

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize preprocessor to disk with joblib."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NYCTripPreprocessor":
        """Load serialized preprocessor from disk."""
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
        return joblib.load(path)
