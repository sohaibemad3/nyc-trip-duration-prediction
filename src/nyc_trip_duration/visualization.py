"""Reusable visualization functions for NYC taxi trip exploratory data analysis."""

from pathlib import Path
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_plotting_style() -> None:
    """Apply consistent publication-ready Matplotlib & Seaborn styling."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
        }
    )


def plot_speed_and_duration(
    df: pd.DataFrame,
    title_suffix: str = "Raw",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot 2x2 grid of speed and duration histograms and boxplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing `real_speed_kmh` and `duration_minutes`.
    title_suffix : str
        Suffix for titles (e.g. 'Raw' or 'Cleaned').
    save_path : Optional[Union[str, Path]]
        Destination path to save figure.

    Returns
    -------
    plt.Figure
    """
    set_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    if "real_speed_kmh" in df.columns:
        sns.histplot(
            df["real_speed_kmh"], bins=30, kde=True, color="#1f4e79", ax=axes[0, 0]
        )
        axes[0, 0].set_title(f"Implied Speed Distribution ({title_suffix})")
        axes[0, 0].set_xlabel("Speed (km/h)")

        sns.boxplot(y=df["real_speed_kmh"], color="#1f4e79", ax=axes[0, 1])
        axes[0, 1].set_title(f"Implied Speed Boxplot ({title_suffix})")
        axes[0, 1].set_ylabel("Speed (km/h)")

    if "duration_minutes" in df.columns:
        sns.histplot(
            df["duration_minutes"], bins=30, kde=True, color="#2e75b6", ax=axes[1, 0]
        )
        axes[1, 0].set_title(f"Trip Duration Distribution ({title_suffix})")
        axes[1, 0].set_xlabel("Duration (minutes)")

        sns.boxplot(y=df["duration_minutes"], color="#2e75b6", ax=axes[1, 1])
        axes[1, 1].set_title(f"Trip Duration Boxplot ({title_suffix})")
        axes[1, 1].set_ylabel("Duration (minutes)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_log_duration_distribution(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot KDE of log1p trip duration highlighting normality.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `log_trip_duration` column.
    save_path : Optional[Union[str, Path]]
        Path to save figure.

    Returns
    -------
    plt.Figure
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(df["log_trip_duration"], color="#1f4e79", fill=True, ax=ax)
    ax.set_title("Log-Transformed Trip Duration Distribution: log(1 + seconds)")
    ax.set_xlabel("log(1 + trip_duration)")
    ax.set_ylabel("Density")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_distance_vs_duration(
    df: pd.DataFrame,
    sample_size: int = 50000,
    random_state: int = 42,
    transformed: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Scatter plot of distance vs log trip duration with optional deterministic subsampling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with distance and log duration.
    sample_size : int, default=50000
        Number of points to sample for rendering performance.
    random_state : int, default=42
        Seed for reproducible sampling.
    transformed : bool, default=False
        Whether distance is power-transformed.
    save_path : Optional[Union[str, Path]]
        Path to save figure.

    Returns
    -------
    plt.Figure
    """
    set_plotting_style()
    plot_df = df
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=random_state)

    fig, ax = plt.subplots(figsize=(7, 5))
    dist_label = (
        "Power-Transformed Distance (Yeo-Johnson)"
        if transformed
        else "Great-Circle Distance (km)"
    )
    sns.scatterplot(
        x="distance_km",
        y="log_trip_duration",
        data=plot_df,
        alpha=0.2,
        s=10,
        color="#2e75b6",
        ax=ax,
    )
    ax.set_title(f"Trip Distance vs Log Duration ({'Transformed' if transformed else 'Raw'})")
    ax.set_xlabel(dist_label)
    ax.set_ylabel("log(1 + trip_duration)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_temporal_patterns(
    df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Plot trip frequency by month, day of month, and hour of day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with temporal features.
    save_dir : Optional[Union[str, Path]]
        Directory to save individual plots.

    Returns
    -------
    Tuple[plt.Figure, plt.Figure, plt.Figure]
        Figures for month, day, and hour.
    """
    set_plotting_style()

    # 1. Trips by month
    fig_month, ax_month = plt.subplots(figsize=(6, 4))
    month_counts = df.groupby("pickup_month")["id"].count().reset_index()
    sns.barplot(data=month_counts, x="pickup_month", y="id", color="#2e75b6", ax=ax_month)
    ax_month.set_title("Trip Count by Month (Jan - Jun 2016)")
    ax_month.set_xlabel("Pickup Month")
    ax_month.set_ylabel("Total Trips")
    if save_dir:
        fig_month.savefig(Path(save_dir) / "trips_by_month.png", bbox_inches="tight")

    # 2. Trips by day of month
    fig_day, ax_day = plt.subplots(figsize=(9, 4))
    day_counts = df.groupby("pickup_day")["id"].count().reset_index()
    sns.barplot(data=day_counts, x="pickup_day", y="id", color="#2e75b6", ax=ax_day)
    ax_day.set_title("Trip Count by Day of Month")
    ax_day.set_xlabel("Day of Month")
    ax_day.set_ylabel("Total Trips")
    if save_dir:
        fig_day.savefig(Path(save_dir) / "trips_by_day.png", bbox_inches="tight")

    # 3. Trips by hour of day
    fig_hour, ax_hour = plt.subplots(figsize=(8, 4))
    hour_counts = df.groupby("pickup_hour")["id"].count().reset_index()
    sns.lineplot(
        data=hour_counts,
        x="pickup_hour",
        y="id",
        marker="o",
        color="#1f4e79",
        ax=ax_hour,
    )
    ax_hour.set_title("Trip Volume by Hour of Day (Rush Hour Profile)")
    ax_hour.set_xlabel("Hour of Day (0-23)")
    ax_hour.set_ylabel("Total Trips")
    ax_hour.set_xticks(range(24))
    if save_dir:
        fig_hour.savefig(Path(save_dir) / "trips_by_hour.png", bbox_inches="tight")

    return fig_month, fig_day, fig_hour
