"""Shared test helpers. Fixtures defined here are available in every test file."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import FAILURE_MODE_COLUMNS, TARGET_COLUMN


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Fake dataset (400 rows, 10% failures). Failures have higher torque, so
    a model can actually learn something from it."""
    rows = 400
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "UDI": range(1, rows + 1),
            "Product ID": [f"L{i}" for i in range(rows)],
            "Type": rng.choice(["L", "M", "H"], size=rows),
            "Air temperature [K]": rng.normal(300, 2, rows).round(1),
            "Process temperature [K]": rng.normal(310, 1.5, rows).round(1),
            "Rotational speed [rpm]": rng.normal(1540, 180, rows).round().astype(int),
            "Torque [Nm]": rng.normal(40, 8, rows).round(1),
            "Tool wear [min]": rng.integers(0, 250, rows),
        }
    )
    failure = np.arange(rows) % 10 == 0
    df[TARGET_COLUMN] = failure.astype(int)
    df.loc[failure, "Torque [Nm]"] += 25  # the signal a model can learn
    for mode in FAILURE_MODE_COLUMNS:
        df[mode] = 0
    return df
