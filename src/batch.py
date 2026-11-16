"""Helpers for the Batch Prediction page (CSV upload)."""

import io

import pandas as pd

from src.predict import InputValidationError

MAX_ROWS = 50_000
RISK_ORDER = ["High", "Medium", "Low"]


def read_csv_bytes(data: bytes, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Read uploaded CSV bytes. Raises InputValidationError with a clear message."""
    try:
        # "utf-8-sig" removes the invisible BOM that Excel/this dataset may add
        table = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise InputValidationError(["The file is empty."]) from None
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise InputValidationError(
            ["The file could not be read as a CSV (check the format and encoding)."]
        ) from None
    table.columns = table.columns.astype(str).str.strip()
    if len(table) > max_rows:
        raise InputValidationError(
            [f"The file has {len(table):,} rows; the limit is {max_rows:,}."]
        )
    return table


def risk_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Number of machines per risk level, always in High / Medium / Low order."""
    counts = result["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    return counts.rename("machines").to_frame()
