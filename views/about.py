"""About page: purpose, method, data source and limitations."""

import streamlit as st

from src.ui import apply_style, hero, load_model_info

apply_style()
hero("About", "What this project is, how it works and what it cannot do.")

info = load_model_info()
metrics = info["test_metrics"]

st.markdown("#### The project")
st.write(
    "This dashboard estimates the failure risk of a machine from six readings "
    "(machine type, air temperature, process temperature, rotational speed, torque "
    "and tool wear). It is built with Python, scikit-learn, XGBoost and Streamlit."
)

st.markdown("#### How it works")
st.markdown(
    "1. The data is cleaned and split 80% / 20% into training and test rows "
    "(stratified, fixed random seed).\n"
    "2. Four models are tuned with 5-fold cross-validation on the training rows only: "
    "Logistic Regression, Decision Tree, Random Forest and XGBoost.\n"
    "3. The final model is chosen by cross-validated PR-AUC **before** the test rows "
    "are used, because failures are rare and accuracy would be misleading.\n"
    "4. The alert threshold is chosen on training data to favour finding failures "
    "(F2 score), since a missed failure usually costs more than a false alarm.\n"
    "5. The test rows are used once, to report honest results."
)

st.markdown("#### The final model")
left, right = st.columns(2)
with left:
    st.markdown(
        f"- **Model:** {info['model_name']}\n"
        f"- **Trained on:** {info['training_rows']:,} rows "
        f"({info['training_failures']} failures)\n"
        f"- **Alert threshold:** {info['threshold']}\n"
        f"- **Saved (UTC):** {info['created_utc']}"
    )
with right:
    st.markdown(
        f"- **Tested on:** {info['test_rows']:,} rows ({info['test_failures']} failures)\n"
        f"- **Failures found:** {metrics['tp']} of {metrics['tp'] + metrics['fn']}\n"
        f"- **False alarms:** {metrics['fp']}\n"
        f"- **PR-AUC:** {metrics['pr_auc']:.3f}"
    )
with st.expander("Library versions used for training"):
    for name, version in info["library_versions"].items():
        st.write(f"**{name}**: {version}")

st.markdown("#### Data source")
st.write(
    "AI4I 2020 Predictive Maintenance Dataset, S. Matzka (2020), UCI Machine "
    "Learning Repository. DOI: 10.24432/C5HS5C. Licence: Creative Commons "
    "Attribution 4.0 International (CC BY 4.0). The data is unmodified."
)

st.markdown("#### Limitations")
st.markdown(
    "- The dataset is **synthetic**; real machines may behave differently.\n"
    f"- The model still misses some failures ({metrics['fn']} of "
    f"{metrics['tp'] + metrics['fn']} in the test rows).\n"
    f"- The test set has only {info['test_failures']} failures, so results can shift "
    "by a few points with a different split.\n"
    "- The failure score is a **risk indicator**, not a guaranteed probability.\n"
    "- The failure rules on the Insights page describe how this dataset's labels "
    "were generated. They are not real-world safety limits.\n"
    "- Random failures (RNF) cannot be predicted from sensor values.\n"
    "- Predictions support inspection decisions; they do not replace them."
)
