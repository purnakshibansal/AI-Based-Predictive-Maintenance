"""Batch Prediction page: upload a CSV and score many machines at once."""

import streamlit as st

from src.batch import read_csv_bytes, risk_summary
from src.data_loader import FEATURE_COLUMNS
from src.predict import (
    InputValidationError,
    predict_dataframe,
    range_warnings,
    validate_features,
)
from src.ui import apply_style, hero, load_cached_model, metric_card

apply_style()
hero(
    "Batch Prediction",
    "Upload a CSV with one row per machine and get a risk level for every row.",
)

model, info = load_cached_model()

st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
st.caption(
    "Extra columns (for example UDI, Product ID or failure flags) are kept in the "
    "output but never given to the model."
)

TEMPLATE = (
    ",".join(FEATURE_COLUMNS)
    + "\nM,298.1,308.6,1551,42.8,0\nL,301.5,310.8,1380,62.0,210\n"
)
st.download_button("Download a template CSV", TEMPLATE, "template.csv", "text/csv")

uploaded = st.file_uploader("Upload your CSV", type="csv")

if uploaded is not None:
    try:
        table = read_csv_bytes(uploaded.getvalue())
        result = predict_dataframe(table, model, info)
        warnings = range_warnings(validate_features(table), info)
    except InputValidationError as error:
        for problem in error.problems:
            st.error(problem)
    else:
        st.success(f"Scored {len(result):,} machines.")
        summary = risk_summary(result)

        cols = st.columns(3)
        with cols[0]:
            metric_card("High risk", str(summary.loc["High", "machines"]), "inspect soon")
        with cols[1]:
            metric_card("Medium risk", str(summary.loc["Medium", "machines"]), "keep an eye on")
        with cols[2]:
            metric_card("Low risk", str(summary.loc["Low", "machines"]), "no action needed")

        for message in warnings:
            st.warning(message)

        st.write("")
        levels = st.multiselect(
            "Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"]
        )
        shown = result[result["risk_level"].isin(levels)].sort_values(
            "failure_score", ascending=False
        )
        st.dataframe(shown, use_container_width=True, hide_index=True)

        st.download_button(
            "Download all results (CSV)",
            result.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )
        st.caption(
            "Scores are risk indicators from a model trained on a synthetic dataset. "
            "They are not guaranteed probabilities."
        )
