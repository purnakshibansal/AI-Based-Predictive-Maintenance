"""Tests for src/predict.py (a small fake model, plus the real saved model)."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import FEATURE_COLUMNS, NUMERIC_FEATURES
from src.predict import (
    INFO_PATH,
    MODEL_PATH,
    InputValidationError,
    load_model,
    predict_dataframe,
    predict_single,
    range_warnings,
    risk_levels,
    validate_features,
    version_warnings,
)
from src.preprocessing import build_preprocessor, prepare_features_and_target

TYPICAL_MACHINE = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
}


@pytest.fixture(scope="module")
def tiny_model(sample_df):
    """A quickly trained model plus a hand-made info dict (threshold 0.5)."""
    features, target = prepare_features_and_target(sample_df)
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(True)),
            ("model", LogisticRegression(class_weight="balanced", max_iter=500)),
        ]
    ).fit(features, target)
    info = {
        "threshold": 0.5,
        "training_ranges": {
            column: {
                "min": float(features[column].min()),
                "max": float(features[column].max()),
            }
            for column in NUMERIC_FEATURES
        },
        "library_versions": {},
    }
    return model, info


def machine(**changes) -> dict:
    return {**TYPICAL_MACHINE, **changes}


def test_single_prediction_has_the_expected_fields(tiny_model):
    model, info = tiny_model
    result = predict_single(TYPICAL_MACHINE, model, info)

    assert 0.0 <= result["failure_score"] <= 1.0
    assert result["predicted_failure"] in (0, 1)
    assert result["risk_level"] in ("Low", "Medium", "High")
    assert result["threshold"] == 0.5
    assert result["warnings"] == []


def test_missing_column_is_reported_by_name(tiny_model):
    model, info = tiny_model
    row = machine()
    del row["Torque [Nm]"]
    with pytest.raises(InputValidationError, match=r"Torque \[Nm\]"):
        predict_single(row, model, info)


def test_unknown_machine_type_is_rejected(tiny_model):
    model, info = tiny_model
    with pytest.raises(InputValidationError, match="Type"):
        predict_single(machine(Type="X"), model, info)


def test_text_in_a_number_column_is_rejected_with_row_number():
    table = pd.DataFrame([machine(), machine(**{"Torque [Nm]": "abc"})])
    with pytest.raises(InputValidationError) as error:
        validate_features(table)
    assert "data row 2" in str(error.value)


def test_impossible_values_are_rejected():
    with pytest.raises(InputValidationError) as error:
        validate_features(
            pd.DataFrame([machine(**{"Tool wear [min]": -5, "Rotational speed [rpm]": 0})])
        )
    assert len(error.value.problems) == 2  # every problem is listed, not just one


def test_empty_input_is_rejected():
    with pytest.raises(InputValidationError, match="no rows"):
        validate_features(pd.DataFrame(columns=FEATURE_COLUMNS))


def test_machine_type_is_cleaned_before_checking(tiny_model):
    model, info = tiny_model
    clean = validate_features(pd.DataFrame([machine(Type=" m ")]))
    assert clean["Type"].tolist() == ["M"]
    assert predict_single(machine(Type=" m "), model, info)["risk_level"]


def test_extra_columns_never_reach_the_model(tiny_model):
    model, info = tiny_model
    plain = pd.DataFrame([TYPICAL_MACHINE])
    with_extras = plain.assign(TWF=1, UDI=99, **{"Machine failure": 1})

    assert list(validate_features(with_extras).columns) == FEATURE_COLUMNS
    same = predict_dataframe(plain, model, info)["failure_score"]
    assert predict_dataframe(with_extras, model, info)["failure_score"].equals(same)


def test_batch_prediction_keeps_rows_and_adds_three_columns(tiny_model):
    model, info = tiny_model
    batch = pd.DataFrame(
        [machine(), machine(**{"Torque [Nm]": 75.0}), machine(Type="L")]
    ).assign(note=["a", "b", "c"])
    result = predict_dataframe(batch, model, info)

    assert len(result) == 3 and result["note"].tolist() == ["a", "b", "c"]
    assert {"failure_score", "predicted_failure", "risk_level"} <= set(result.columns)
    assert (result["predicted_failure"] == (result["failure_score"] >= 0.5)).all()


def test_risk_level_rule():
    import numpy as np

    levels = risk_levels(np.array([0.05, 0.15, 0.5]), 0.2)
    assert levels.tolist() == ["Low", "Medium", "High"]


def test_values_outside_the_training_range_give_a_warning(tiny_model):
    model, info = tiny_model
    result = predict_single(machine(**{"Torque [Nm]": 500.0}), model, info)
    assert any("Torque [Nm]" in text for text in result["warnings"])
    clean = validate_features(pd.DataFrame([TYPICAL_MACHINE]))
    assert range_warnings(clean, info) == []


def test_library_version_mismatch_is_reported():
    assert version_warnings({"library_versions": {"scikit-learn": "0.0.1"}})
    assert version_warnings({"library_versions": {}}) == []


def test_missing_model_file_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="finalize_model"):
        load_model(tmp_path / "nope.joblib", tmp_path / "nope.json")


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and INFO_PATH.is_file()),
    reason="final model not saved yet (run python -m src.finalize_model)",
)
def test_the_real_saved_model_can_predict():
    model, info = load_model()
    result = predict_single(TYPICAL_MACHINE, model, info)
    assert 0.0 <= result["failure_score"] <= 1.0
    assert info["features"] == FEATURE_COLUMNS
