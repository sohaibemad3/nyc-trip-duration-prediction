"""Model builders, serialization, and factory registry."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from nyc_trip_duration.config import (
    RIDGE_CONFIG,
    XGB_CONFIG,
    MLP_CONFIG,
    MODELS_DIR,
)


def build_ridge(
    alpha: Optional[float] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Ridge:
    """Instantiate a Ridge regression model with project defaults.

    Parameters
    ----------
    alpha : Optional[float], default=1.0
        Regularization strength.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    **kwargs : Any
        Additional arguments passed to Ridge.

    Returns
    -------
    Ridge
        Configured Ridge estimator.
    """
    config = dict(RIDGE_CONFIG)
    if alpha is not None:
        config["alpha"] = alpha
    if random_state is not None:
        config["random_state"] = random_state
    config.update(kwargs)
    return Ridge(**config)


def build_xgboost(
    n_estimators: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> XGBRegressor:
    """Instantiate an XGBoost Regressor matching the historical configuration.

    Parameters
    ----------
    n_estimators : Optional[int], default=200
        Number of gradient boosted trees.
    learning_rate : Optional[float], default=0.05
        Boosting learning rate (eta).
    max_depth : Optional[int], default=5
        Maximum tree depth for base learners.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    **kwargs : Any
        Additional arguments passed to XGBRegressor.

    Returns
    -------
    XGBRegressor
        Configured XGBoost estimator.
    """
    config = dict(XGB_CONFIG)
    if n_estimators is not None:
        config["n_estimators"] = n_estimators
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if max_depth is not None:
        config["max_depth"] = max_depth
    if random_state is not None:
        config["random_state"] = random_state
    config.update(kwargs)
    return XGBRegressor(**config)


def build_mlp(
    hidden_layer_sizes: Optional[tuple] = None,
    max_iter: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> MLPRegressor:
    """Instantiate an MLP Regressor matching the historical configuration.

    Parameters
    ----------
    hidden_layer_sizes : Optional[tuple], default=(64, 32)
        Architecture of hidden layers.
    max_iter : Optional[int], default=1000
        Maximum number of iterations.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    **kwargs : Any
        Additional arguments passed to MLPRegressor.

    Returns
    -------
    MLPRegressor
        Configured MLP estimator.
    """
    config = dict(MLP_CONFIG)
    if hidden_layer_sizes is not None:
        config["hidden_layer_sizes"] = hidden_layer_sizes
    if max_iter is not None:
        config["max_iter"] = max_iter
    if random_state is not None:
        config["random_state"] = random_state
    config.update(kwargs)
    return MLPRegressor(**config)


def build_model(model_name: str, **kwargs: Any) -> Any:
    """Model factory creating an estimator by canonical name.

    Parameters
    ----------
    model_name : str
        One of 'ridge', 'xgboost' (or 'xgb'), 'mlp' (or 'neural_network').
    **kwargs : Any
        Hyperparameter overrides passed to builder.

    Returns
    -------
    Estimator
        Configured model instance.
    """
    normalized = model_name.lower().strip()
    if normalized in ["ridge", "linear"]:
        return build_ridge(**kwargs)
    elif normalized in ["xgb", "xgboost", "xgbregressor"]:
        return build_xgboost(**kwargs)
    elif normalized in ["mlp", "nn", "neural_network", "mlpregressor"]:
        return build_mlp(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported models: 'ridge', 'xgboost', 'mlp'."
        )


def save_model(model: Any, filepath: Union[str, Path]) -> Path:
    """Save model artifact to disk with joblib."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath: Union[str, Path]) -> Any:
    """Load model artifact from disk."""
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)
