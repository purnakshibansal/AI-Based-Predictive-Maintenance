"""Maintenance Insights page: what the data shows and a rule check per machine."""

import plotly.express as px
import streamlit as st

from src.data_loader import load_raw_data
from src.eda import failure_mode_summary, outlier_report, type_summary
from src.predict import InputValidationError, predict_single
from src.preprocessing import MACHINE_TYPES
from src.recommendations import check_failure_rules, power_watts
from src.ui import apply_style, hero, load_cached_model


@st.cache_data
def load_data():
    return load_raw_data()


apply_style()
hero(
    "Maintenance Insights",
    "What the data shows, and which documented failure rules a machine triggers.",
)

try:
    data = load_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.markdown("#### What the data shows")
left, right = st.columns(2)
with left:
    types = type_summary(data).reset_index()
    figure = px.bar(
        types, x="Type", y="failure_rate_percent", text="failure_rate_percent",
        title="Failure rate per machine type (%)",
        color_discrete_sequence=["#D1495B"],
    )
    st.plotly_chart(figure, width="stretch")
with right:
    modes = failure_mode_summary(data).reset_index()
    figure = px.bar(
        modes, x="mode", y="rows_flagged", text="rows_flagged",
        title="Rows flagged per failure mode",
        color_discrete_sequence=["#F59E0B"],
    )
    st.plotly_chart(figure, width="stretch")

torque = outlier_report(data).loc["Torque [Nm]"]
st.info(
    f"Rows with unusual torque ({int(torque['outliers'])} rows) failed "
    f"{torque['failure_rate_in_outliers_%']}% of the time, against "
    f"{torque['failure_rate_elsewhere_%']}% for all other rows. "
    "Unusual readings are strong warning signs."
)
st.caption(
    "RNF (random failure) cannot be predicted from sensor values: "
    "it was assigned at random in the dataset."
)

st.markdown("#### Check a machine")
model, info = load_cached_model()

with st.form("insights_form"):
    a, b = st.columns(2)
    with a:
        machine_type = st.selectbox("Machine type", MACHINE_TYPES, index=1)
        air = st.number_input("Air temperature [K]", min_value=1.0, value=298.1, step=0.1)
        process = st.number_input("Process temperature [K]", min_value=1.0, value=308.6, step=0.1)
    with b:
        speed = st.number_input("Rotational speed [rpm]", min_value=1.0, value=1551.0, step=10.0)
        torque_value = st.number_input("Torque [Nm]", min_value=0.0, value=42.8, step=0.5)
        wear = st.number_input("Tool wear [min]", min_value=0.0, value=0.0, step=1.0)
    submitted = st.form_submit_button("Check machine", type="primary")

if submitted:
    values = {
        "Type": machine_type,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque_value,
        "Tool wear [min]": wear,
    }
    try:
        result = predict_single(values, model, info)
    except InputValidationError as error:
        for problem in error.problems:
            st.error(problem)
    else:
        level = result["risk_level"]
        message = f"Model risk level: **{level}** (failure score {result['failure_score']:.2f})."
        if level == "High":
            st.error(message)
        elif level == "Medium":
            st.warning(message)
        else:
            st.success(message)

        st.caption(
            f"Derived values: temperature difference {process - air:.1f} K, "
            f"power about {power_watts(torque_value, speed):,.0f} W, "
            f"tool wear x torque {wear * torque_value:,.0f} minNm."
        )

        rules = check_failure_rules(values)
        if rules:
            st.markdown("**Documented failure rules triggered**")
            for rule in rules:
                st.warning(f"**{rule['title']}** ({rule['mode']}): {rule['why']}\n\n"
                           f"Suggested action: {rule['action']}")
        else:
            st.success(
                "None of the documented failure rules is triggered. "
                "The model can still flag a risk from combinations of readings."
            )

st.caption(
    "These rules describe how the synthetic AI4I 2020 failure labels were generated. "
    "They are not real-world safety limits. Suggested actions are general hints."
)
