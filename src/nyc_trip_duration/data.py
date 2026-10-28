"""Data loading, path resolution, and storage utilities."""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from nyc_trip_duration.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    TRAIN_FILENAME,
    VAL_FILENAME,
    TEST_FILENAME,
    RANDOM_SEED,
)


def resolve_data_path(filename_or_path: Union[str, Path]) -> Path:
    """Resolve the location of a dataset file.

    Checks:
    1. Direct path as given (absolute or relative)
    2. data/raw/<filename>
    3. project_root/<filename>

    Parameters
    ----------
    filename_or_path : Union[str, Path]
        Filename or relative/absolute path.

    Returns
    -------
    Path
        Resolved existing path.

    Raises
    ------
    FileNotFoundError
        If file cannot be located in any supported location.
    """
    candidate = Path(filename_or_path)
    if candidate.is_file():
        return candidate.resolve()

    raw_candidate = RAW_DATA_DIR / candidate.name
    if raw_candidate.is_file():
        return raw_candidate.resolve()

    root_candidate = PROJECT_ROOT / candidate.name
    if root_candidate.is_file():
        return root_candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate dataset '{filename_or_path}'. Checked locations:\n"
        f" - {candidate.resolve()}\n"
        f" - {raw_candidate.resolve()}\n"
        f" - {root_candidate.resolve()}\n"
        f"Please verify data placement per data/README.md instructions."
    )


def load_dataset(
    split: str = "train",
    nrows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load a dataset split ('train', 'val', or 'test') with optional subsampling.

    Parameters
    ----------
    split : str, default='train'
        One of 'train', 'val', or 'test'.
    nrows : Optional[int], default=None
        Maximum number of rows to read directly from the CSV.
    sample_fraction : Optional[float], default=None
        Deterministic random fraction of rows to sample after reading.
    random_state : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    split_map = {
        "train": TRAIN_FILENAME,
        "val": VAL_FILENAME,
        "validation": VAL_FILENAME,
        "test": TEST_FILENAME,
    }
    key = split.lower().strip()
    if key not in split_map:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of: 'train', 'val', 'test'."
        )

    file_path = resolve_data_path(split_map[key])
    df = pd.read_csv(file_path, nrows=nrows)

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Save a DataFrame to disk, creating parent directories if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    output_path : Union[str, Path]
        Target destination path (.csv or .parquet).

    Returns
    -------
    Path
        Absolute path of saved file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    return path
