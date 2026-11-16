"""Tests for src/evaluate_model.py (small fake data, tiny models)."""

import json

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate_model import (
    choose_threshold,
    compute_metrics,
    evaluate_models,
    out_of_fold_probabilities,
    plot_confusion_matrices,
    plot_precision_recall_curves,
    plot_roc_curves,
    save_metrics,
    select_final_model,
)
from src.preprocessing import (
    build_preprocessor,
    prepare_features_and_target,
    split_data,
)


@pytest.fixture(scope="module")
def small_setup(sample_df):
    features, target = prepare_features_and_target(sample_df)
    x_train, x_test, y_train, y_test = split_data(features, target)
    models = {
        "Logistic Regression": Pipeline(
            [("preprocess", build_preprocessor(True)),
             ("model", LogisticRegression(class_weight="balanced", max_iter=500))]
        ).fit(x_train, y_train),
        "Decision Tree": Pipeline(
            [("preprocess", build_preprocessor(False)),
             ("model", DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                              random_state=0))]
        ).fit(x_train, y_train),
    }
    return models, x_train, x_test, y_train, y_test


def test_compute_metrics_on_a_hand_checked_example():
    # predictions at threshold 0.5 are [0, 1, 0, 1] -> one of each outcome
    result = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5 and result["recall"] == 0.5


def test_metrics_handle_a_model_that_never_predicts_failure():
    result = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], 0.9)
    assert result["tp"] == 0 and result["precision"] == 0.0 and result["f1"] == 0.0


def test_choose_threshold_balances_catching_failures_and_false_alarms():
    y_true = [0, 0, 0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.3, 0.6, 0.5, 0.9]
    # At 0.5 both failures are found with one false alarm (best F2 score).
    assert choose_threshold(y_true, probabilities) == 0.5


def test_choose_threshold_needs_at_least_one_failure():
    with pytest.raises(ValueError):
        choose_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_out_of_fold_probabilities_cover_every_training_row(small_setup):
    models, x_train, _, y_train, _ = small_setup
    probabilities = out_of_fold_probabilities(models["Logistic Regression"], x_train, y_train)
    assert probabilities.shape == (len(x_train),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_evaluate_models_returns_all_metrics(small_setup):
    models, x_train, x_test, y_train, y_test = small_setup
    metrics, probabilities = evaluate_models(models, x_train, y_train, x_test, y_test)

    assert list(metrics) == list(models)
    for name, m in metrics.items():
        assert m["tp"] + m["fn"] == int(y_test.sum())  # every real failure counted
        assert m["tp"] + m["fn"] + m["fp"] + m["tn"] == len(y_test)
        assert 0 <= m["threshold"] <= 1
        assert len(probabilities[name]) == len(y_test)


def test_final_model_is_chosen_from_cv_results_only(tmp_path):
    path = tmp_path / "cv.json"
    path.write_text(json.dumps([
        {"model": "A", "pr_auc": 0.5},
        {"model": "B", "pr_auc": 0.7},
        {"model": "C", "pr_auc": 0.6},
    ]), encoding="utf-8")
    name, reason = select_final_model(path)
    assert name == "B" and "0.700" in reason


def test_report_and_figures_are_written(small_setup, tmp_path):
    models, x_train, x_test, y_train, y_test = small_setup
    metrics, probabilities = evaluate_models(models, x_train, y_train, x_test, y_test)

    report_path = save_metrics(metrics, "Decision Tree", "because", len(y_test),
                               int(y_test.sum()), tmp_path / "metrics.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selected_model"] == "Decision Tree"
    assert set(report["models"]) == set(models)

    figures = [
        plot_confusion_matrices(metrics, tmp_path),
        plot_precision_recall_curves(y_test, probabilities, metrics, tmp_path),
        plot_roc_curves(y_test, probabilities, metrics, tmp_path),
    ]
    for figure in figures:
        assert figure.exists() and figure.stat().st_size > 1000
