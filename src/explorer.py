"""Helpers for the Dataset Explorer page."""

import pandas as pd

from src.data_loader import TARGET_COLUMN

STATUS_LABELS = {0: "normal", 1: "failure"}


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a readable 'Status' column (normal / failure)."""
    table = df.copy()
    table["Status"] = table[TARGET_COLUMN].map(STATUS_LABELS)
    return table


def filter_data(
    df: pd.DataFrame, machine_types: list[str], statuses: list[str]
) -> pd.DataFrame:
    """Keep only rows with the chosen machine types and statuses."""
    mask = df["Type"].isin(machine_types) & df["Status"].isin(statuses)
    return df[mask]
