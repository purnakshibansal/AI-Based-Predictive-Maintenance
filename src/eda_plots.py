"""EDA charts for the AI4I 2020 dataset, saved as PNG files.

The numbers behind each chart come from src/eda.py. Run everything with:

    python -m src.eda_plots

Charts are saved in reports/figures/. No data is removed to make a chart
look nicer: outliers stay in every plot.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import (
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_raw_data,
)
from src.eda import class_balance, failure_mode_summary, outlier_report, type_summary

matplotlib.use("Agg")  # draw straight to files; no screen is needed (cloud-safe)
sns.set_theme(style="whitegrid")

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
NORMAL_COLOR = "#2E86AB"  # steel blue
FAILURE_COLOR = "#D1495B"  # red
STATUS_ORDER = ["normal", "failure"]
STATUS_PALETTE = {"normal": NORMAL_COLOR, "failure": FAILURE_COLOR}


def _save(fig: plt.Figure, name: str, folder: Path) -> Path:
    """Save a figure as PNG, close it, and return the file path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """Bar chart: normal rows vs failure rows (shows the class imbalance)."""
    table = class_balance(df)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        ["Normal (0)", "Failure (1)"],
        table["count"],
        color=[NORMAL_COLOR, FAILURE_COLOR],
    )
    for bar, count, percent in zip(bars, table["count"], table["percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({percent}%)",
            ha="center",
            va="bottom",
        )
    ax.set_title("Machine failure distribution (class imbalance)")
    ax.set_ylabel("Number of rows")
    ax.set_ylim(0, table["count"].max() * 1.15)
    return _save(fig, "01_class_distribution.png", folder)


def plot_type_distribution(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """Two bar charts: machines per type, and failure rate per type."""
    table = type_summary(df)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars = left.bar(table.index, table["machines"], color=NORMAL_COLOR)
    left.bar_label(bars, fmt="{:,.0f}")
    left.set_title("Machines per type")
    left.set_xlabel("Machine type (L = low, M = medium, H = high quality)")
    left.set_ylabel("Number of machines")
    left.set_ylim(0, table["machines"].max() * 1.12)

    bars = right.bar(table.index, table["failure_rate_percent"], color=FAILURE_COLOR)
    right.bar_label(bars, fmt="{:.2f}%")
    right.set_title("Failure rate per type")
    right.set_xlabel("Machine type")
    right.set_ylabel("Failure rate (%)")
    right.set_ylim(0, table["failure_rate_percent"].max() * 1.2)
    return _save(fig, "02_machine_type_distribution.png", folder)


def plot_failure_modes(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """Bar chart: how many rows are flagged for each failure mode."""
    table = failure_mode_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(table.index, table["rows_flagged"], color=FAILURE_COLOR)
    ax.bar_label(bars, fmt="{:,.0f}")
    ax.set_title("Failure modes (rows flagged per mode)")
    ax.set_xlabel("TWF = tool wear, HDF = heat dissipation, PWF = power, "
                  "OSF = overstrain, RNF = random")
    ax.set_ylabel("Number of rows")
    ax.set_ylim(0, table["rows_flagged"].max() * 1.15)
    return _save(fig, "03_failure_modes.png", folder)


def plot_sensor_histograms(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """One histogram (with density curve) per sensor column."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat_axes = axes.flatten()
    for ax, column in zip(flat_axes, NUMERIC_FEATURES):
        sns.histplot(data=df, x=column, bins=40, kde=True, color=NORMAL_COLOR, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_ylabel("Number of rows")
    flat_axes[-1].set_visible(False)  # 5 sensors, 6 slots: hide the empty one
    fig.suptitle("Sensor value distributions", fontsize=14)
    return _save(fig, "04_sensor_histograms.png", folder)


def plot_sensor_boxplots(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """One boxplot per sensor. Dots beyond the whiskers are the outliers."""
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4.8))
    for ax, column in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, y=column, color=NORMAL_COLOR, ax=ax)
        ax.set_title(column, fontsize=10)
        ax.set_ylabel("")
    fig.suptitle("Sensor boxplots (outliers are kept, shown as dots)", fontsize=14)
    return _save(fig, "05_sensor_boxplots.png", folder)


def plot_correlation_heatmap(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """Heatmap of correlations between the sensors and the failure label."""
    corr = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax
    )
    ax.set_title("Correlation heatmap (sensors and machine failure)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "06_correlation_heatmap.png", folder)


def plot_sensors_by_failure(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """Boxplots comparing each sensor for normal rows vs failure rows."""
    data = df[NUMERIC_FEATURES].copy()
    data["Status"] = df[TARGET_COLUMN].map({0: "normal", 1: "failure"})

    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4.8))
    for ax, column in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(
            data=data,
            x="Status",
            y=column,
            hue="Status",
            order=STATUS_ORDER,
            hue_order=STATUS_ORDER,
            palette=STATUS_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Sensor readings: normal vs failure", fontsize=14)
    return _save(fig, "07_sensors_by_failure.png", folder)


def plot_outlier_failure_rates(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> Path:
    """Failure rate inside outlier readings vs everywhere else (IQR rule)."""
    table = outlier_report(df)
    table = table[table["outliers"] > 0]  # only sensors that have outliers

    labels = [
        f"{name}\n({int(row['outliers'])} outlier rows)"
        for name, row in table.iterrows()
    ]
    positions = list(range(len(table)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    in_outliers = ax.bar(
        [p - width / 2 for p in positions],
        table["failure_rate_in_outliers_%"],
        width,
        color=FAILURE_COLOR,
        label="Rows with an outlier reading",
    )
    elsewhere = ax.bar(
        [p + width / 2 for p in positions],
        table["failure_rate_elsewhere_%"],
        width,
        color=NORMAL_COLOR,
        label="All other rows",
    )
    ax.bar_label(in_outliers, fmt="{:.1f}%")
    ax.bar_label(elsewhere, fmt="{:.1f}%")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Outlier readings are strong warning signs (so we keep them)")
    ax.set_ylim(0, 105)
    ax.legend()
    return _save(fig, "08_outlier_failure_rates.png", folder)


def save_all_figures(df: pd.DataFrame, folder: Path = FIGURES_DIR) -> list[Path]:
    """Create every EDA chart and return the list of saved file paths."""
    return [
        plot_class_distribution(df, folder),
        plot_type_distribution(df, folder),
        plot_failure_modes(df, folder),
        plot_sensor_histograms(df, folder),
        plot_sensor_boxplots(df, folder),
        plot_correlation_heatmap(df, folder),
        plot_sensors_by_failure(df, folder),
        plot_outlier_failure_rates(df, folder),
    ]


def main() -> None:
    """Load the dataset and save all EDA charts."""
    paths = save_all_figures(load_raw_data())
    print(f"Saved {len(paths)} figures in {FIGURES_DIR}:")
    for path in paths:
        print(" -", path.name)


if __name__ == "__main__":
    main()
