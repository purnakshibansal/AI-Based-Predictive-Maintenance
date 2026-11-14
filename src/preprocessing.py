"""Prepare the AI4I 2020 data for machine learning.

Main ideas:
* Only ``Type`` and the five sensor columns are used as model inputs.
* Encoding and scaling live inside a scikit-learn ColumnTransformer. Later it
  is combined with a model into ONE Pipeline, so training and prediction
  always apply exactly the same steps.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import (
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

RANDOM_STATE = 42  # fixed seed so every run gives the same split and results
TEST_SIZE = 0.2  # 20% of rows are kept aside for the final evaluation
MACHINE_TYPES = ["L", "M", "H"]  # the three product quality variants

# Columns that must NEVER be given to a model as input (leakage / identifiers).
FORBIDDEN_FEATURES = [TARGET_COLUMN, *FAILURE_MODE_COLUMNS, *ID_COLUMNS]


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the raw table into model inputs (X) and the target (y)."""
    # Safety net: stops us if someone edits FEATURE_COLUMNS by mistake.
    leaked = [column for column in FEATURE_COLUMNS if column in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"Forbidden columns used as features: {leaked}")

    features = df[FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].copy()
    return features, target


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Create the preprocessing step (not yet fitted).

    * ``Type`` -> one-hot encoding (3 columns of 0/1). The categories are fixed,
      so an unknown machine type raises an error instead of being ignored.
    * Sensor columns -> StandardScaler when ``scale_numeric`` is True (needed by
      Logistic Regression), otherwise left unchanged (tree models do not need it).

    The output is a plain NumPy array on purpose: XGBoost does not accept column
    names that contain square brackets, such as "Torque [Nm]".
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "machine_type",
                OneHotEncoder(categories=[MACHINE_TYPES], sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("sensors", numeric_step, NUMERIC_FEATURES),
        ]
    )


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split.

    Stratified means failures are split in the same proportion (about 3.4%)
    into both parts, which matters because failures are rare.
    Returns: X_train, X_test, y_train, y_test
    """
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
