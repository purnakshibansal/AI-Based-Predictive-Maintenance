"""Tests for src/eda.py, using a tiny hand-made table (6 machines)."""

import pandas as pd

from src.eda import (
    class_balance,
    label_consistency_report,
    outlier_report,
    type_summary,
)


def make_tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3, 4, 5, 6],
            "Product ID": ["L1", "L2", "M3", "M4", "H5", "H6"],
            "Type": ["L", "L", "M", "M", "H", "H"],
            "Air temperature [K]": [300.0] * 6,
            "Process temperature [K]": [310.0] * 6,
            "Rotational speed [rpm]": [1500, 1500, 1500, 1500, 1500, 3000],
            "Torque [Nm]": [40.0] * 6,
            "Tool wear [min]": [10, 20, 30, 40, 50, 60],
            "Machine failure": [0, 1, 0, 1, 0, 0],
            "TWF": [0, 1, 0, 0, 0, 0],
            "HDF": [0, 0, 0, 0, 0, 0],
            "PWF": [0, 0, 0, 0, 0, 0],
            "OSF": [0, 0, 0, 0, 0, 0],
            "RNF": [0, 0, 0, 0, 0, 1],
        }
    )


def test_class_balance_counts_and_percent():
    table = class_balance(make_tiny_df())
    assert table["count"].tolist() == [4, 2]
    assert table["percent"].tolist() == [66.67, 33.33]


def test_type_summary_failure_rates():
    table = type_summary(make_tiny_df())
    assert list(table.index) == ["L", "M", "H"]
    assert table["failures"].tolist() == [1, 1, 0]
    assert table["failure_rate_percent"].tolist() == [50.0, 50.0, 0.0]


def test_label_consistency_report():
    report = label_consistency_report(make_tiny_df())
    assert report["failures"] == 2
    assert report["failures_with_no_mode_flag"] == 1  # machine 4
    assert report["mode_flagged_but_no_failure"] == 1  # machine 6 (RNF)
    assert report["rows_with_two_or_more_modes"] == 0


def test_outlier_report_counts_but_never_removes_rows():
    df = make_tiny_df()
    report = outlier_report(df)
    assert len(df) == 6  # nothing was removed
    assert report.loc["Rotational speed [rpm]", "outliers"] == 1  # the 3000 rpm row
    assert report.loc["Tool wear [min]", "outliers"] == 0
