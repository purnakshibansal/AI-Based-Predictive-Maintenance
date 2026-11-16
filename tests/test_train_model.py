"""Tests for src/train_model.py (uses a small fake dataset, not the real CSV)."""

import joblib
import pandas as pd
import pytest

from src.preprocessing import prepare_features_and_target, split_data
from src.train_model import (
    SCORING,
    SELECTION_METRIC,
    compute_scale_pos_weight,
    save_candidates,
    save_cv_summary,
    summarize_cv,
    train_all_models,
)

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


@pytest.fixture(scope="module")
def trained(sample_df):
    """Train all four models once and share the result between the tests."""
    features, target = prepare_features_and_target(sample_df)
    x_train, x_test, y_train, _ = split_data(features, target)
    searches = train_all_models(x_train, y_train, cv_splits=3, n_jobs=1)
    return searches, x_test


def test_scale_pos_weight_is_normal_divided_by_failures():
    target = pd.Series([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(target) == 9.0


def test_scale_pos_weight_rejects_data_without_failures():
    with pytest.raises(ValueError):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))


def test_all_four_models_are_trained_and_give_probabilities(trained):
    searches, x_test = trained
    assert list(searches) == MODEL_NAMES
    for search in searches.values():
        probabilities = search.best_estimator_.predict_proba(x_test)
        assert probabilities.shape == (len(x_test), 2)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_summary_is_sorted_and_models_learned_something(trained):
    searches, _ = trained
    summary = summarize_cv(searches)

    for metric in SCORING:
        assert metric in summary.columns
    scores = summary[SELECTION_METRIC].tolist()
    assert scores == sorted(scores, reverse=True)
    assert not summary[list(SCORING)].isna().any().any()  # every score was computed
    assert summary[SELECTION_METRIC].min() > 0.3  # base rate is only 0.10


def test_saved_models_reload_and_predict_the_same(trained, tmp_path):
    searches, x_test = trained
    paths = save_candidates(searches, tmp_path)
    assert len(paths) == 4

    original = searches["Random Forest"].best_estimator_.predict_proba(x_test)
    reloaded = joblib.load(tmp_path / "random_forest.joblib").predict_proba(x_test)
    assert (original == reloaded).all()


def test_cv_summary_json_is_written(trained, tmp_path):
    searches, _ = trained
    path = save_cv_summary(summarize_cv(searches), tmp_path / "cv.json")
    assert path.exists() and "pr_auc" in path.read_text(encoding="utf-8")
