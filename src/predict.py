"""Load the saved model and make predictions (Phase 9).

The dashboard (single prediction and CSV upload) will call the functions here.

Important wording: the model gives a "failure score" between 0 and 1. Because
failures are rare and we used class weights, this score is NOT a guaranteed
real-world probability. It is a risk indicator, nothing more.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS, NUMERIC_FEATURES, PROJECT_ROOT
from src.preprocessing import MACHINE_TYPES

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"

# Basic physical sense checks. We do NOT invent upper safety limits.
MUST_BE_POSITIVE = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
]
MUST_BE_NON_NEGATIVE = ["Torque [Nm]", "Tool wear [min]"]


class InputValidationError(ValueError):
    """Raised when the input data cannot be used. `problems` lists every issue."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def load_model(model_path: Path = MODEL_PATH, info_path: Path = INFO_PATH):
    """Load the saved pipeline (preprocessing + model) and its info file."""
    model_path, info_path = Path(model_path), Path(info_path)
    for path in (model_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python -m src.finalize_model` first."
            )
    model = joblib.load(model_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return model, info


def version_warnings(info: dict) -> list[str]:
    """Warn if the libraries now installed differ from the ones used for training."""
    import sklearn
    import xgboost

    installed = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
    saved = info.get("library_versions", {})
    return [
        f"{name} is {installed[name]} here but the model was saved with "
        f"{saved[name]}; predictions may differ or fail."
        for name in installed
        if name in saved and saved[name] != installed[name]
    ]


def _first_rows(mask: pd.Series, limit: int = 5) -> str:
    """Describe the first few offending rows, e.g. 'data rows 3, 7'."""
    rows = [str(position + 1) for position in np.flatnonzero(mask.to_numpy())[:limit]]
    more = ", ..." if int(mask.sum()) > limit else ""
    return "data row" + ("s " if len(rows) > 1 else " ") + ", ".join(rows) + more


def validate_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the input and return a clean table with exactly the model inputs.

    Extra columns (for example UDI, Product ID or TWF) are ignored, so they can
    never reach the model. Raises InputValidationError listing every problem.
    """
    if raw.empty:
        raise InputValidationError(["The input contains no rows."])

    missing = [column for column in FEATURE_COLUMNS if column not in raw.columns]
    if missing:
        raise InputValidationError([f"Missing required columns: {missing}"])

    problems: list[str] = []
    clean = pd.DataFrame(index=raw.index)

    machine_type = raw["Type"].astype(str).str.strip().str.upper()
    bad_type = ~machine_type.isin(MACHINE_TYPES)
    if bad_type.any():
        problems.append(
            f"Type must be one of {MACHINE_TYPES} ({_first_rows(bad_type)} invalid)."
        )
    clean["Type"] = machine_type

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(raw[column], errors="coerce").astype(float)
        unusable = ~np.isfinite(values)
        if unusable.any():
            problems.append(
                f"{column}: {int(unusable.sum())} value(s) are missing or not "
                f"numbers ({_first_rows(unusable)})."
            )
        if column in MUST_BE_POSITIVE:
            wrong = (values <= 0) & ~unusable
            rule = "greater than 0"
        else:
            wrong = (values < 0) & ~unusable
            rule = "0 or more"
        if wrong.any():
            problems.append(f"{column} must be {rule} ({_first_rows(wrong)}).")
        clean[column] = values

    if problems:
        raise InputValidationError(problems)
    return clean[FEATURE_COLUMNS]


def range_warnings(clean: pd.DataFrame, info: dict) -> list[str]:
    """Warn about values outside the range the model saw during training."""
    warnings = []
    for column, limits in info.get("training_ranges", {}).items():
        outside = (clean[column] < limits["min"]) | (clean[column] > limits["max"])
        if outside.any():
            warnings.append(
                f"{int(outside.sum())} row(s) have {column} outside the range seen "
                f"in training ({limits['min']} to {limits['max']}); the model is "
                "less reliable there."
            )
    return warnings


def risk_levels(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Simple display rule: High if score >= threshold, Medium if >= half of it."""
    return np.select(
        [scores >= threshold, scores >= threshold / 2], ["High", "Medium"], "Low"
    )


def predict_dataframe(raw: pd.DataFrame, model, info: dict) -> pd.DataFrame:
    """Predict for many rows. Returns the input columns plus three new ones:
    failure_score, predicted_failure (0/1) and risk_level."""
    clean = validate_features(raw)
    scores = model.predict_proba(clean)[:, 1]
    threshold = info["threshold"]

    result = raw.reset_index(drop=True).copy()
    result["failure_score"] = np.round(scores, 4)
    result["predicted_failure"] = (scores >= threshold).astype(int)
    result["risk_level"] = risk_levels(scores, threshold)
    return result


def predict_single(values: dict, model, info: dict) -> dict:
    """Predict for one machine, given a dict of the six input values."""
    row = pd.DataFrame([values])
    result = predict_dataframe(row, model, info).iloc[0]
    return {
        "failure_score": float(result["failure_score"]),
        "predicted_failure": int(result["predicted_failure"]),
        "risk_level": str(result["risk_level"]),
        "threshold": float(info["threshold"]),
        "warnings": range_warnings(validate_features(row), info),
    }
