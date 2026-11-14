"""Load and validate the AI4I 2020 predictive maintenance dataset.

This module is the single place that knows the dataset's column names.
Other modules import the column lists from here, so a typo can only
happen in one location.
"""

from pathlib import Path

import pandas as pd

# Project root = the folder that contains `src/`, `data/`, `app.py`, ...
# Building the path from __file__ means it works wherever Python is started
# (Codespaces terminal, pytest, or Streamlit Community Cloud).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

# --- Column names, copied exactly from the CSV header ---
ID_COLUMNS = ["UDI", "Product ID"]  # identifiers, never used as features
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # model inputs

TARGET_COLUMN = "Machine failure"  # what we predict (0 = normal, 1 = failure)

# Failure-mode flags describe HOW a machine failed. They are only known after
# the failure, so using them as inputs would be target leakage.
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

EXPECTED_COLUMNS = (
    ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS
)


def find_missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return the names from `required` that are not columns of `df`."""
    return [column for column in required if column not in df.columns]


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that all expected columns are present.

    The data itself is not changed in any way.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Upload ai4i2020.csv into the data/raw/ folder."
        )

    # "utf-8-sig" removes the invisible BOM character at the start of the file,
    # which would otherwise turn the first column name into "\ufeffUDI".
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    missing = find_missing_columns(df, EXPECTED_COLUMNS)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, int | float]:
    """Return a few basic facts about the dataset (rows, missing, failures...)."""
    failures = int(df[TARGET_COLUMN].sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "failures": failures,
        "normal": int(df.shape[0]) - failures,
        "failure_rate_percent": round(100 * failures / df.shape[0], 2),
    }
