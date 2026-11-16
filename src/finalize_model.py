"""Save the selected model as the single file the dashboard loads (Phase 9).

Reads the choice made in Phase 8 (reports/model_metrics.json), copies that
tuned pipeline to models/best_model.joblib, writes models/model_info.json,
then reloads the saved file and checks that it reproduces the Phase 8 results.

Run it with:

    python -m src.finalize_model
"""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS, NUMERIC_FEATURES, load_raw_data
from src.evaluate_model import METRICS_PATH, candidate_path, compute_metrics
from src.predict import INFO_PATH, MODEL_PATH, load_model
from src.preprocessing import (
    MACHINE_TYPES,
    RANDOM_STATE,
    prepare_features_and_target,
    split_data,
)
from src.train_model import CANDIDATES_DIR


def library_versions() -> dict[str, str]:
    """Versions of the libraries the saved model depends on."""
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def build_model_info(
    model_name: str, report: dict, features_train: pd.DataFrame, target_train
) -> dict:
    """Collect everything a user or the dashboard should know about the model."""
    scores = report["models"][model_name]
    ranges = {
        column: {
            "min": float(features_train[column].min()),
            "max": float(features_train[column].max()),
        }
        for column in NUMERIC_FEATURES
    }
    return {
        "model_name": model_name,
        "selection_reason": report["selection_reason"],
        "threshold": scores["threshold"],
        "threshold_rule": report["threshold_rule"],
        "features": FEATURE_COLUMNS,
        "machine_types": MACHINE_TYPES,
        "random_state": RANDOM_STATE,
        "training_rows": len(features_train),
        "training_failures": int(target_train.sum()),
        "training_ranges": ranges,
        "test_rows": report["test_rows"],
        "test_failures": report["test_failures"],
        "test_metrics": scores,
        "library_versions": library_versions(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": [
            "The failure score is a risk indicator, not a guaranteed probability.",
            "The model was trained on a synthetic dataset (AI4I 2020).",
            "The model was trained on the training rows only; test rows were held out.",
        ],
    }


def finalize_model(
    features_train: pd.DataFrame,
    target_train,
    report_path: Path = METRICS_PATH,
    candidates_dir: Path = CANDIDATES_DIR,
    model_path: Path = MODEL_PATH,
    info_path: Path = INFO_PATH,
) -> dict:
    """Save the selected pipeline and its info file. Returns the info dict."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    model_name = report["selected_model"]
    pipeline = joblib.load(candidate_path(model_name, Path(candidates_dir)))

    info = build_model_info(model_name, report, features_train, target_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    Path(info_path).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def verify_saved_model(
    x_test: pd.DataFrame,
    y_test,
    report_path: Path = METRICS_PATH,
    model_path: Path = MODEL_PATH,
    info_path: Path = INFO_PATH,
) -> dict[str, int]:
    """Reload the saved files and confirm they reproduce the Phase 8 counts."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    model, info = load_model(model_path, info_path)

    probabilities = model.predict_proba(x_test)[:, 1]
    now = compute_metrics(y_test, probabilities, info["threshold"])
    expected = report["models"][info["model_name"]]

    counts = {key: now[key] for key in ("tp", "fn", "fp", "tn")}
    for key, value in counts.items():
        if value != expected[key]:
            raise RuntimeError(
                f"Saved model does not reproduce Phase 8: {key} is {value}, "
                f"expected {expected[key]}."
            )
    return counts


def main() -> None:
    """Save the final model, then verify it by reloading it from disk."""
    features, target = prepare_features_and_target(load_raw_data())
    x_train, x_test, y_train, y_test = split_data(features, target)

    info = finalize_model(x_train, y_train)
    print(f"Saved final model: {info['model_name']}")
    print(f"  file: {MODEL_PATH}")
    print(f"  size: {MODEL_PATH.stat().st_size / 1024:.0f} KB")
    print(f"  threshold: {info['threshold']}")

    counts = verify_saved_model(x_test, y_test)
    print(f"Reloaded from disk and re-tested: {counts}")
    print("The counts match the Phase 8 report.")


if __name__ == "__main__":
    main()
