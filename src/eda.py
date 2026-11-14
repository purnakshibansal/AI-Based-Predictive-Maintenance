"""Exploratory Data Analysis (EDA) helpers for the AI4I 2020 dataset.

Every function here only READS the data and returns a small summary table.
Nothing is deleted, filled in or changed. Run the whole report with:

    python -m src.eda
"""

import pandas as pd

from src.data_loader import (
    FAILURE_MODE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_raw_data,
)
from src.preprocessing import MACHINE_TYPES


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Data type, non-missing count, missing count and unique values per column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "unique_values": df.nunique(),
        }
    )


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """How many values are missing in each column (count and percent)."""
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "missing": missing,
            "missing_percent": (100 * missing / len(df)).round(2),
        }
    )


def duplicate_report(df: pd.DataFrame) -> dict[str, int]:
    """Count duplicated rows, duplicated IDs and repeated measurements."""
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_UDI": int(df["UDI"].duplicated().sum()),
        "duplicate_Product_ID": int(df["Product ID"].duplicated().sum()),
        "same_measurements_ignoring_ids": int(
            df.drop(columns=ID_COLUMNS).duplicated().sum()
        ),
    }


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """How many normal rows and how many failure rows (the class imbalance)."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    table = pd.DataFrame(
        {"count": counts, "percent": (100 * counts / len(df)).round(2)}
    )
    table.index = table.index.map({0: "0 = normal", 1: "1 = failure"})
    return table


def type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Machines and failure rate for each machine type (L, M, H)."""
    table = df.groupby("Type")[TARGET_COLUMN].agg(machines="count", failures="sum")
    table["failure_rate_percent"] = (
        100 * table["failures"] / table["machines"]
    ).round(2)
    return table.reindex(MACHINE_TYPES)


def failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows flagged for each failure mode, and how many of them are not failures."""
    rows = []
    for mode in FAILURE_MODE_COLUMNS:
        flagged = df[mode] == 1
        rows.append(
            {
                "mode": mode,
                "rows_flagged": int(flagged.sum()),
                "flagged_but_no_machine_failure": int(
                    (flagged & (df[TARGET_COLUMN] == 0)).sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("mode")


def label_consistency_report(df: pd.DataFrame) -> dict[str, int]:
    """Check how the failure-mode flags relate to the 'Machine failure' label."""
    mode_count = df[FAILURE_MODE_COLUMNS].sum(axis=1)
    failure = df[TARGET_COLUMN] == 1
    return {
        "failures": int(failure.sum()),
        "failures_with_no_mode_flag": int((failure & (mode_count == 0)).sum()),
        "mode_flagged_but_no_failure": int((~failure & (mode_count > 0)).sum()),
        "rows_with_two_or_more_modes": int((mode_count >= 2).sum()),
    }


def sensor_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Count, mean, std, min, quartiles and max of each sensor column."""
    return df[NUMERIC_FEATURES].describe().T.round(2)


def sensors_by_failure(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and std of each sensor, for normal rows vs failure rows."""
    stats = (
        df.groupby(TARGET_COLUMN)[NUMERIC_FEATURES]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    stats.index = stats.index.map({0: "normal", 1: "failure"})
    return stats.T


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Find unusual readings with the IQR rule. The rows are only counted, never removed.

    A value is flagged when it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    """
    rows = []
    for column in NUMERIC_FEATURES:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = (df[column] < low) | (df[column] > high)

        def failure_rate(mask: pd.Series) -> float:
            return round(100 * df.loc[mask, TARGET_COLUMN].mean(), 2) if mask.any() else 0.0

        rows.append(
            {
                "sensor": column,
                "lower_limit": round(low, 2),
                "upper_limit": round(high, 2),
                "outliers": int(is_outlier.sum()),
                "outlier_percent": round(100 * is_outlier.mean(), 2),
                "failure_rate_in_outliers_%": failure_rate(is_outlier),
                "failure_rate_elsewhere_%": failure_rate(~is_outlier),
            }
        )
    return pd.DataFrame(rows).set_index("sensor")


def main() -> None:
    """Print the full numeric EDA report."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    df = load_raw_data()
    sections = [
        ("1. Dataset size (rows, columns)", df.shape),
        ("2. First five rows", df.head()),
        ("3. Column information", column_info(df)),
        ("4. Missing values", missing_value_report(df)),
        ("5. Duplicates", duplicate_report(df)),
        ("6. Class balance (target)", class_balance(df)),
        ("7. Machine types", type_summary(df)),
        ("8. Failure modes", failure_mode_summary(df)),
        ("9. Label consistency", label_consistency_report(df)),
        ("10. Sensor statistics", sensor_statistics(df)),
        ("11. Sensors: normal vs failure", sensors_by_failure(df)),
        ("12. Outliers (IQR rule, nothing removed)", outlier_report(df)),
    ]
    for title, content in sections:
        print(f"\n=== {title} ===")
        print(content)


if __name__ == "__main__":
    main()
