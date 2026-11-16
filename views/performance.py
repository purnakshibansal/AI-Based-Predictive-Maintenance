"""Model Performance page: test-set results of the four models."""

import json

import pandas as pd
import streamlit as st

from src.data_loader import PROJECT_ROOT
from src.ui import apply_style, hero, metric_card

REPORT_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

COLUMNS = [
    "pr_auc", "roc_auc", "recall", "precision", "f1",
    "balanced_accuracy", "accuracy", "threshold", "tp", "fn", "fp", "tn",
]


@st.cache_data
def load_report() -> dict:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            "reports/model_metrics.json not found. Run `python -m src.evaluate_model`."
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def show_figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chart {name} not found. Run `python -m src.evaluate_model`.")


apply_style()
hero(
    "Model Performance",
    "How the four models did on test rows they never saw during training.",
)

report = load_report()
selected = report["selected_model"]
best = report["models"][selected]

st.subheader(f"Selected model: {selected}")
st.write(report["selection_reason"])
st.caption(
    f"Test set: {report['test_rows']:,} rows with {report['test_failures']} real failures. "
    f"Decision threshold rule: {report['threshold_rule']}."
)

cols = st.columns(4)
with cols[0]:
    metric_card("Recall", f"{best['recall']:.1%}", "share of real failures found")
with cols[1]:
    metric_card("Precision", f"{best['precision']:.1%}", "share of alarms that are real")
with cols[2]:
    metric_card("PR-AUC", f"{best['pr_auc']:.3f}", "main ranking metric")
with cols[3]:
    metric_card("ROC-AUC", f"{best['roc_auc']:.3f}", "overall ranking quality")

st.write("")
st.markdown("#### All models compared")
table = pd.DataFrame(report["models"]).T[COLUMNS].sort_values("pr_auc", ascending=False)
table.insert(0, "selected", ["yes" if name == selected else "" for name in table.index])
for column in ("tp", "fn", "fp", "tn"):
    table[column] = table[column].astype(int)
st.dataframe(table, width="stretch")
st.caption(
    "tp = failures found, fn = failures missed, fp = false alarms, "
    "tn = normal rows correctly left alone."
)

st.warning(
    "Accuracy looks high for every model because 96.6% of rows are normal. "
    "A model that always says 'no failure' would score the same. "
    "Use recall, precision and PR-AUC instead."
)

st.markdown("#### Charts")
tab1, tab2, tab3 = st.tabs(["Confusion matrices", "Precision-recall", "ROC"])
with tab1:
    show_figure("09_confusion_matrices.png", "Counts on the test rows")
with tab2:
    show_figure("10_precision_recall_curves.png", "Higher and further right is better")
with tab3:
    show_figure("11_roc_curves.png", "Closer to the top-left corner is better")

st.markdown("#### Limitations")
st.markdown(
    "- The test set has only 68 failures, so a few failures more or less change recall "
    "by about 1.5 points.\n"
    "- The model still misses some failures, so it supports inspection decisions "
    "and does not replace them.\n"
    "- The dataset is synthetic; real machines may behave differently."
)
