"""Evaluate the four trained models on the held-out test rows (Phase 8).

Rules that keep the evaluation honest:
* The final model is chosen BEFORE looking at the test rows, using the
  cross-validation PR-AUC from training (reports/cv_results.json).
* Each model's decision threshold is chosen from out-of-fold predictions on
  the TRAINING rows only (never from the test rows).
* The 2,000 test rows are used once, to report how every model performs.

Run it with:

    python -m src.evaluate_model
"""

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_loader import PROJECT_ROOT, load_raw_data
from src.preprocessing import RANDOM_STATE, prepare_features_and_target, split_data
from src.train_model import CANDIDATES_DIR, CV_RESULTS_PATH, SELECTION_METRIC

matplotlib.use("Agg")  # draw straight to files (cloud-safe)

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Threshold rule: maximise F2, which counts recall twice as much as precision.
# Reason: for maintenance a missed failure (false negative) usually costs more
# than an unnecessary inspection (false alarm). This is a project decision
# that can be changed here (beta = 1.0 would weight both equally).
THRESHOLD_BETA = 2.0
CV_FOLDS = 5

MODEL_COLORS = {
    "Logistic Regression": "#8D99AE",
    "Decision Tree": "#EDAE49",
    "Random Forest": "#2E86AB",
    "XGBoost": "#D1495B",
}


def candidate_path(name: str, folder: Path = CANDIDATES_DIR) -> Path:
    """File that train_model.py used when saving this model."""
    return folder / f"{name.lower().replace(' ', '_')}.joblib"


def load_candidates(folder: Path = CANDIDATES_DIR) -> dict:
    """Load the four tuned pipelines (preprocessing + model) from disk."""
    models = {}
    for name in MODEL_NAMES:
        path = candidate_path(name, folder)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python -m src.train_model` first."
            )
        models[name] = joblib.load(path)
    return models


def out_of_fold_probabilities(pipeline, features_train, target_train) -> np.ndarray:
    """Failure probability for every TRAINING row, predicted by a model that did
    not see that row (5-fold cross-validation). Used to choose the threshold."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(
        pipeline, features_train, target_train, cv=cv, method="predict_proba"
    )[:, 1]


def choose_threshold(y_true, probabilities, beta: float = THRESHOLD_BETA) -> float:
    """Probability cut-off that gives the best F-beta score on the given rows.

    A row is predicted as "failure" when its probability is >= the threshold.
    """
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        raise ValueError("Cannot choose a threshold: there are no failures (1s).")

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision, recall = precision[:-1], recall[:-1]  # align with `thresholds`
    numerator = (1 + beta**2) * precision * recall
    denominator = beta**2 * precision + recall
    scores = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return float(thresholds[int(np.argmax(scores))])


def compute_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    """All evaluation numbers for one model at one threshold."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    def safe_divide(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(safe_divide(tp + tn, len(y_true)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predicted)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(safe_divide(2 * precision * recall, precision + recall), 4),
        "f2": round(
            safe_divide(5 * precision * recall, 4 * precision + recall), 4
        ),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_models(models: dict, x_train, y_train, x_test, y_test):
    """Return (metrics, test_probabilities) for every model.

    Threshold: from training rows only. Metrics: from the test rows.
    """
    metrics, test_probabilities = {}, {}
    for name, pipeline in models.items():
        train_probabilities = out_of_fold_probabilities(pipeline, x_train, y_train)
        threshold = choose_threshold(y_train, train_probabilities)

        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics[name] = compute_metrics(y_test, probabilities, threshold)
        test_probabilities[name] = probabilities
    return metrics, test_probabilities


def select_final_model(cv_results_path: Path = CV_RESULTS_PATH) -> tuple[str, str]:
    """Pick the model with the best cross-validation PR-AUC (training data only).

    Returns (model name, plain-language reason).
    """
    records = json.loads(Path(cv_results_path).read_text(encoding="utf-8"))
    best = max(records, key=lambda record: record[SELECTION_METRIC])
    reason = (
        f"Highest cross-validated PR-AUC ({best[SELECTION_METRIC]:.3f}) on the "
        "training data. PR-AUC suits rare failures better than accuracy. The "
        "choice was made before the test rows were used."
    )
    return best["model"], reason


def save_metrics(
    metrics: dict,
    selected_model: str,
    reason: str,
    test_rows: int,
    test_failures: int,
    path: Path = METRICS_PATH,
) -> Path:
    """Write the evaluation results as JSON (the dashboard will read this file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "selected_model": selected_model,
        "selection_reason": reason,
        "threshold_rule": f"maximise F{THRESHOLD_BETA:g} on out-of-fold training predictions",
        "test_rows": int(test_rows),
        "test_failures": int(test_failures),
        "models": metrics,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _save(fig: plt.Figure, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrices(metrics: dict, folder: Path = FIGURES_DIR) -> Path:
    """One confusion matrix per model (test rows, tuned threshold)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
    for ax, (name, m) in zip(np.atleast_1d(axes), metrics.items()):
        matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                ax.text(
                    col, row, f"{value:,}", ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black",
                    fontsize=13,
                )
        ax.set_xticks([0, 1], ["Normal", "Failure"])
        ax.set_yticks([0, 1], ["Normal", "Failure"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nthreshold {m['threshold']:.2f}", fontsize=10)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test rows", fontsize=13)
    return _save(fig, "09_confusion_matrices.png", folder)


def plot_precision_recall_curves(
    y_test, test_probabilities: dict, metrics: dict, folder: Path = FIGURES_DIR
) -> Path:
    """Precision-recall curve of every model on the test rows."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, probabilities in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        ax.plot(
            recall, precision, color=MODEL_COLORS.get(name),
            label=f"{name} (PR-AUC {metrics[name]['pr_auc']:.3f})", linewidth=2,
        )
    ax.axhline(
        np.mean(y_test), color="grey", linestyle="--", linewidth=1,
        label="Always guessing failure (baseline)",
    )
    ax.set_xlabel("Recall (share of real failures found)")
    ax.set_ylabel("Precision (share of alarms that are real)")
    ax.set_title("Precision-recall curves on the test rows")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, "10_precision_recall_curves.png", folder)


def plot_roc_curves(
    y_test, test_probabilities: dict, metrics: dict, folder: Path = FIGURES_DIR
) -> Path:
    """ROC curve of every model on the test rows."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, probabilities in test_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        ax.plot(
            false_positive_rate, true_positive_rate, color=MODEL_COLORS.get(name),
            label=f"{name} (ROC-AUC {metrics[name]['roc_auc']:.3f})", linewidth=2,
        )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test rows")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "11_roc_curves.png", folder)


def main() -> None:
    """Evaluate all models, save metrics and charts, and print a summary."""
    features, target = prepare_features_and_target(load_raw_data())
    x_train, x_test, y_train, y_test = split_data(features, target)

    selected, reason = select_final_model()
    print(f"Selected before seeing the test rows: {selected}")

    models = load_candidates()
    metrics, test_probabilities = evaluate_models(models, x_train, y_train, x_test, y_test)

    save_metrics(metrics, selected, reason, len(y_test), int(y_test.sum()))
    plot_confusion_matrices(metrics)
    plot_precision_recall_curves(y_test, test_probabilities, metrics)
    plot_roc_curves(y_test, test_probabilities, metrics)

    pd.set_option("display.width", 160)
    table = pd.DataFrame(metrics).T
    print(f"\nTest rows: {len(y_test)} ({int(y_test.sum())} failures)")
    print("\nTest-set results (threshold chosen on training data):")
    print(table[["threshold", "accuracy", "balanced_accuracy", "precision",
                 "recall", "f1", "roc_auc", "pr_auc"]].to_string())
    print("\nConfusion counts (missed failures = fn, false alarms = fp):")
    print(table[["tp", "fn", "fp", "tn"]].astype(int).to_string())

    test_ranking = table["pr_auc"].sort_values(ascending=False).index.tolist()
    print(f"\nTest PR-AUC ranking: {test_ranking}")
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
