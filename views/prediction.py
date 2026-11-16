"""Prediction page: enter one machine's readings and get a risk level."""

import streamlit as st

from src.predict import InputValidationError, predict_single, version_warnings
from src.preprocessing import MACHINE_TYPES
from src.ui import apply_style, hero, load_cached_model, metric_card

apply_style()
hero(
    "Failure Risk Prediction",
    "Enter the six machine readings and check the estimated failure risk.",
)

model, info = load_cached_model()
for message in version_warnings(info):
    st.warning(message)

ranges = info["training_ranges"]
st.caption(
    "Default values are one typical machine from the dataset. "
    "Values outside the range seen in training still work, but you get a warning."
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        machine_type = st.selectbox(
            "Machine type (L = low, M = medium, H = high quality)",
            MACHINE_TYPES,
            index=1,
        )
        air = st.number_input("Air temperature [K]", min_value=1.0, value=298.1, step=0.1)
        process = st.number_input("Process temperature [K]", min_value=1.0, value=308.6, step=0.1)
    with right:
        speed = st.number_input("Rotational speed [rpm]", min_value=1.0, value=1551.0, step=10.0)
        torque = st.number_input("Torque [Nm]", min_value=0.0, value=42.8, step=0.5)
        wear = st.number_input("Tool wear [min]", min_value=0.0, value=0.0, step=1.0)
    submitted = st.form_submit_button("Predict failure risk", type="primary")

with st.expander("Value ranges seen in training"):
    for column, limits in ranges.items():
        st.write(f"**{column}**: {limits['min']} to {limits['max']}")

if submitted:
    values = {
        "Type": machine_type,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
    }
    try:
        result = predict_single(values, model, info)
    except InputValidationError as error:
        for problem in error.problems:
            st.error(problem)
    else:
        level = result["risk_level"]
        if level == "High":
            st.error("**High risk**: inspect this machine soon.")
        elif level == "Medium":
            st.warning("**Medium risk**: keep an eye on this machine.")
        else:
            st.success("**Low risk** based on these readings.")

        cols = st.columns(3)
        with cols[0]:
            metric_card("Failure score", f"{result['failure_score']:.2f}", "risk indicator, not a probability")
        with cols[1]:
            metric_card("Model verdict", "Failure" if result["predicted_failure"] else "No failure",
                        f"alarm threshold {result['threshold']:.2f}")
        with cols[2]:
            metric_card("Risk level", level, "Low / Medium / High")

        for message in result["warnings"]:
            st.warning(message)
        st.caption(
            "This is a risk indicator from a model trained on a synthetic dataset. "
            "It misses some real failures, so it supports inspection decisions "
            "and does not replace them."
        )
