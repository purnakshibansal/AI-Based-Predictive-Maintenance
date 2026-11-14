"""Tests for src/preprocessing.py.

They use a small made-up table, so they do not need the real CSV file.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessing import (
    FORBIDDEN_FEATURES,
    build_preprocessor,
    prepare_features_and_target,
    split_data,
)


def make_sample_df(rows: int = 200) -> pd.DataFrame:
    """Build a fake dataset with the same columns as the real one (10% failures)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "UDI": range(1, rows + 1),
            "Product ID": [f"M{14860 + i}" for i in range(rows)],
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
    return df


def test_features_exclude_forbidden_columns():
    features, target = prepare_features_and_target(make_sample_df())
    assert list(features.columns) == FEATURE_COLUMNS
    assert not set(features.columns) & set(FORBIDDEN_FEATURES)
    assert target.name == TARGET_COLUMN


def test_leakage_guard_raises_error(monkeypatch):
    monkeypatch.setattr("src.preprocessing.FEATURE_COLUMNS", FEATURE_COLUMNS + ["TWF"])
    with pytest.raises(ValueError, match="Forbidden"):
        prepare_features_and_target(make_sample_df())


def test_split_is_stratified_and_reproducible():
    features, target = prepare_features_and_target(make_sample_df())
    x_train, x_test, y_train, y_test = split_data(features, target)
    again = split_data(features, target)

    assert len(x_test) == 40 and len(x_train) == 160
    assert y_train.sum() == 16 and y_test.sum() == 4  # same 10% in both parts
    assert not set(x_train.index) & set(x_test.index)  # no row in both parts
    assert list(x_train.index) == list(again[0].index)  # same seed, same split


def test_preprocessor_with_scaling():
    features, _ = prepare_features_and_target(make_sample_df())
    output = build_preprocessor(scale_numeric=True).fit_transform(features)

    assert output.shape == (200, 3 + len(NUMERIC_FEATURES))
    assert np.allclose(output[:, :3].sum(axis=1), 1)  # exactly one Type per row
    assert np.allclose(output[:, 3:].mean(axis=0), 0, atol=1e-8)  # scaled


def test_preprocessor_without_scaling_keeps_values():
    features, _ = prepare_features_and_target(make_sample_df())
    output = build_preprocessor(scale_numeric=False).fit_transform(features)
    assert np.allclose(output[:, 3:], features[NUMERIC_FEATURES].to_numpy())


def test_scaler_learns_only_from_training_data():
    features, target = prepare_features_and_target(make_sample_df())
    x_train, x_test, _, _ = split_data(features, target)
    preprocessor = build_preprocessor(scale_numeric=True).fit(x_train)
    learned_means = preprocessor.named_transformers_["sensors"].mean_
    assert np.allclose(learned_means, x_train[NUMERIC_FEATURES].mean().to_numpy())
    assert preprocessor.transform(x_test).shape[0] == len(x_test)


def test_unknown_machine_type_is_rejected():
    features, _ = prepare_features_and_target(make_sample_df())
    preprocessor = build_preprocessor(scale_numeric=True).fit(features)
    bad_row = features.head(1).copy()
    bad_row["Type"] = "X"
    with pytest.raises(ValueError):
        preprocessor.transform(bad_row)
