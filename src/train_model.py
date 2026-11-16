"""Train and tune the four candidate models (Phase 7).

Models: Logistic Regression, Decision Tree, Random Forest, XGBoost.

How the training stays honest:
* Only the TRAINING part of the data is used here. The test part is never
  touched until the evaluation step (src/evaluate_model.py).
* Each model is tuned with 5-fold cross-validation INSIDE the training data.
* Failures are rare (about 3.4%), so every model is told to pay extra
  attention to them (class weights / scale_pos_weight).
* The main score for choosing settings is PR-AUC (average precision), which
  suits rare events better than accuracy.

Run it with:

    python -m src.train_model
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_loader import PROJECT_ROOT, load_raw_data
from src.preprocessing import (
    RANDOM_STATE,
    build_preprocessor,
    prepare_features_and_target,
    split_data,
)

CANDIDATES_DIR = PROJECT_ROOT / "models" / "candidates"
CV_RESULTS_PATH = PROJECT_ROOT / "reports" / "cv_results.json"

# Scores measured during cross-validation (all on held-out folds of the
# training data). "refit" below uses PR-AUC to pick the best settings.
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}
SELECTION_METRIC = "pr_auc"


def compute_scale_pos_weight(target: pd.Series) -> float:
    """Ratio normal : failure, used to make XGBoost take failures seriously."""
    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    if positives == 0:
        raise ValueError("The training target contains no failures (no 1 values).")
    return negatives / positives


def _pipeline(scale_numeric: bool, model) -> Pipeline:
    """Preprocessing + model in ONE object, so both always travel together."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def build_candidates(target_train: pd.Series) -> dict[str, tuple[Pipeline, dict]]:
    """Return {model name: (pipeline, settings to try)} for the four models."""
    return {
        "Logistic Regression": (
            _pipeline(
                True,  # this model needs scaled sensor values
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            _pipeline(
                False,  # trees do not need scaling
                DecisionTreeClassifier(
                    class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
            {
                "model__max_depth": [3, 5, 7, 10],
                "model__min_samples_leaf": [5, 20, 50],
            },
        ),
        "Random Forest": (
            _pipeline(
                False,
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
            {
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5, 20],
            },
        ),
        "XGBoost": (
            _pipeline(
                False,
                XGBClassifier(
                    scale_pos_weight=compute_scale_pos_weight(target_train),
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
            {
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__n_estimators": [200, 400],
            },
        ),
    }


def train_all_models(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    cv_splits: int = 5,
    n_jobs: int = -1,
) -> dict[str, GridSearchCV]:
    """Tune each model with cross-validation and refit the best version."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    searches: dict[str, GridSearchCV] = {}

    for name, (pipeline, grid) in build_candidates(target_train).items():
        started = time.time()
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=SCORING,
            refit=SELECTION_METRIC,
            cv=cv,
            n_jobs=n_jobs,
        )
        search.fit(features_train, target_train)
        searches[name] = search
        print(f"  trained {name:<20} in {time.time() - started:5.1f} s")
    return searches


def summarize_cv(searches: dict[str, GridSearchCV]) -> pd.DataFrame:
    """One row per model: cross-validated scores of its best settings."""
    rows = []
    for name, search in searches.items():
        best = search.best_index_
        row = {"model": name}
        for metric in SCORING:
            row[metric] = round(float(search.cv_results_[f"mean_test_{metric}"][best]), 4)
        row["pr_auc_std"] = round(float(search.cv_results_["std_test_pr_auc"][best]), 4)
        row["best_params"] = {
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model")
    return summary.sort_values(SELECTION_METRIC, ascending=False)


def save_candidates(
    searches: dict[str, GridSearchCV], folder: Path = CANDIDATES_DIR
) -> list[Path]:
    """Save each tuned pipeline (preprocessing + model) with joblib."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, search in searches.items():
        path = folder / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(search.best_estimator_, path)
        paths.append(path)
    return paths


def save_cv_summary(summary: pd.DataFrame, path: Path = CV_RESULTS_PATH) -> Path:
    """Write the cross-validation summary as a small JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = summary.reset_index().to_dict(orient="records")
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    """Load data, split it, train all four models and save the results."""
    features, target = prepare_features_and_target(load_raw_data())
    x_train, _x_test, y_train, _y_test = split_data(features, target)
    print(f"Training on {len(x_train)} rows ({int(y_train.sum())} failures).")
    print("The test rows are NOT used here.\n")

    searches = train_all_models(x_train, y_train)

    summary = summarize_cv(searches)
    print("\nCross-validation results (mean over 5 folds, training data only):")
    print(summary.drop(columns="best_params").to_string())
    print("\nBest settings found:")
    for name, params in summary["best_params"].items():
        print(f"  {name}: {params}")

    save_candidates(searches)
    save_cv_summary(summary)
    print(f"\nSaved models in {CANDIDATES_DIR}")
    print(f"Saved summary in {CV_RESULTS_PATH}")


if __name__ == "__main__":
    main()
