"""Home page: what the project is and how well the final model did."""

import streamlit as st

from src.ui import apply_style, hero, load_model_info, metric_card

apply_style()
hero(
    "AI-Based Predictive Maintenance",
    "Estimate machine failure risk from sensor readings (AI4I 2020 dataset).",
)

info = load_model_info()
m = info["test_metrics"]

st.subheader(f"Final model: {info['model_name']}")
st.caption(
    f"Scores below come from {info['test_rows']:,} held-out test rows "
    f"({info['test_failures']} real failures) that the model never saw in training."
)

cols = st.columns(4)
with cols[0]:
    metric_card("Failures found", f"{m['tp']} / {m['tp'] + m['fn']}",
                f"recall {m['recall']:.1%}")
with cols[1]:
    metric_card("Alarm precision", f"{m['precision']:.1%}",
                f"{m['fp']} false alarms")
with cols[2]:
    metric_card("PR-AUC", f"{m['pr_auc']:.3f}", "main ranking metric")
with cols[3]:
    metric_card("Failures missed", str(m["fn"]), "the main weakness")

st.write("")
left, right = st.columns(2)
with left:
    st.markdown("#### How to use this dashboard")
    st.markdown(
        "- Enter the six machine readings to get a risk level.\n"
        "- Upload a CSV to score many machines at once.\n"
        "- Explore the data and model results on the other pages."
    )
with right:
    st.markdown("#### Please read")
    st.warning(
        "The failure score is a **risk indicator**, not a guaranteed probability. "
        "The model still misses some failures, and the dataset is synthetic."
    )
