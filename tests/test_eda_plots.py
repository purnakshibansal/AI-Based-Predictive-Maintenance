"""Tests for src/eda_plots.py. Charts are written to a temporary folder."""

import numpy as np
import pandas as pd

from src.data_loader import FAILURE_MODE_COLUMNS, TARGET_COLUMN
from src.eda_plots import save_all_figures


def make_sample_df(rows: int = 300) -> pd.DataFrame:
    """Fake dataset with the real column names (10% failures, some variation)."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "UDI": range(1, rows + 1),
            "Product ID": [f"L{i}" for i in range(rows)],
            "Type": rng.choice(["L", "M", "H"], size=rows),
            "Air temperature [K]": rng.normal(300, 2, rows).round(1),
            "Process temperature [K]": rng.normal(310, 1.5, rows).round(1),
            "Rotational speed [rpm]": rng.normal(1540, 180, rows).round().astype(int),
            "Torque [Nm]": rng.normal(40, 10, rows).round(1),
            "Tool wear [min]": rng.integers(0, 250, rows),
        }
    )
    df[TARGET_COLUMN] = [1 if i % 10 == 0 else 0 for i in range(rows)]
    for mode in FAILURE_MODE_COLUMNS:
        df[mode] = 0
    df.loc[df[TARGET_COLUMN] == 1, "HDF"] = 1
    # Two extreme torque readings, so the outlier chart has something to show.
    df.loc[[5, 6], "Torque [Nm]"] = [120.0, 2.0]
    return df


def test_save_all_figures_creates_eight_png_files(tmp_path):
    paths = save_all_figures(make_sample_df(), tmp_path)

    assert len(paths) == 8
    for path in paths:
        assert path.exists() and path.suffix == ".png"
        assert path.stat().st_size > 1000  # a real image, not an empty file
    assert (tmp_path / "01_class_distribution.png").exists()
    assert (tmp_path / "08_outlier_failure_rates.png").exists()


def test_plotting_does_not_change_the_data(tmp_path):
    df = make_sample_df()
    before = df.copy()
    save_all_figures(df, tmp_path)
    pd.testing.assert_frame_equal(df, before)  # no rows or values were touched
