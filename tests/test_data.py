"""Unit tests for data loading, resolution, and persistence."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from nyc_trip_duration.data import (
    resolve_data_path,
    load_dataset,
    save_dataset,
)


def test_resolve_existing_file(tmp_path: Path):
    """Test resolving a file path directly."""
    test_file = tmp_path / "dummy.csv"
    test_file.write_text("a,b\n1,2\n")
    resolved = resolve_data_path(test_file)
    assert resolved == test_file.resolve()


def test_resolve_missing_file_raises():
    """Test that resolving a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_data_path("non_existent_file_xyz_123.csv")
    assert "Could not locate dataset" in str(exc_info.value)


def test_load_dataset_invalid_split():
    """Test loading an unknown split raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        load_dataset("invalid_split_name")
    assert "Unknown split" in str(exc_info.value)


def test_save_and_reload_dataset(tmp_path: Path):
    """Test persisting and re-loading a DataFrame."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    out_file = tmp_path / "subdir" / "saved.csv"

    saved_path = save_dataset(df, out_file)
    assert saved_path.is_file()

    reloaded = pd.read_csv(saved_path)
    pd.testing.assert_frame_equal(df, reloaded)
