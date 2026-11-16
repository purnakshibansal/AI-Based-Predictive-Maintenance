"""Tests for src/finalize_model.py (small fake data, temporary folders)."""

import json

import joblib
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate_model import candidate_path, compute_metrics
from src.finalize_model import build_model_info, finalize_model, verify_saved_model
from src.predict import load_model
from src.preprocessing import (
    build_preprocessor,
    prepare_features_and_target,
    split_data,
)


@pytest.fixture
def workspace(sample_df, tmp_path):
    """A temporary candidates folder, a Phase 8 style report and output paths."""
    features, target = prepare_features_and_target(sample_df)
    x_train, x_test, y_train, y_test = split_data(features, target)
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(False)),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ]
    ).fit(x_train, y_train)

    candidates = tmp_path / "candidates"
    candidates.mkdir()
    joblib.dump(pipeline, candidate_path("Decision Tree", candidates))

    metrics = compute_metrics(y_test, pipeline.predict_proba(x_test)[:, 1], 0.5)
    report = {
        "selected_model": "Decision Tree",
        "selection_reason": "test reason",
        "threshold_rule": "test rule",
        "test_rows": len(y_test),
        "test_failures": int(y_test.sum()),
        "models": {"Decision Tree": metrics},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
        "candidates": candidates, "report_path": report_path,
        "model_path": tmp_path / "models" / "best_model.joblib",
        "info_path": tmp_path / "models" / "model_info.json",
        "report": report,
    }


def run_finalize(w):
    return finalize_model(
        w["x_train"], w["y_train"], w["report_path"], w["candidates"],
        w["model_path"], w["info_path"],
    )


def test_info_contains_the_key_facts(workspace):
    info = build_model_info("Decision Tree", workspace["report"],
                            workspace["x_train"], workspace["y_train"])
    assert info["threshold"] == 0.5
    assert info["training_rows"] == len(workspace["x_train"])
    assert set(info["training_ranges"]) == {
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]",
    }
    assert {"python", "scikit-learn", "xgboost"} <= set(info["library_versions"])


def test_finalize_saves_files_that_reload(workspace):
    info = run_finalize(workspace)
    assert workspace["model_path"].is_file() and workspace["info_path"].is_file()

    model, saved_info = load_model(workspace["model_path"], workspace["info_path"])
    assert saved_info["model_name"] == info["model_name"] == "Decision Tree"
    assert model.predict_proba(workspace["x_test"]).shape[1] == 2


def test_verification_passes_when_the_saved_model_matches_the_report(workspace):
    run_finalize(workspace)
    counts = verify_saved_model(
        workspace["x_test"], workspace["y_test"], workspace["report_path"],
        workspace["model_path"], workspace["info_path"],
    )
    assert sum(counts.values()) == len(workspace["y_test"])


def test_verification_fails_when_the_report_does_not_match(workspace):
    run_finalize(workspace)
    report = workspace["report"]
    report["models"]["Decision Tree"]["tp"] += 1  # tamper with the expected result
    workspace["report_path"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not reproduce"):
        verify_saved_model(
            workspace["x_test"], workspace["y_test"], workspace["report_path"],
            workspace["model_path"], workspace["info_path"],
        )
