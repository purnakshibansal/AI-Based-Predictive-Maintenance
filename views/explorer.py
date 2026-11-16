"""Dataset Explorer page: look at the raw data and its patterns."""

import plotly.express as px
import streamlit as st

from src.data_loader import (
    FAILURE_MODE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_raw_data,
)
from src.explorer import STATUS_LABELS, add_status, filter_data
from src.preprocessing import MACHINE_TYPES
from src.ui import apply_style, hero, metric_card

COLORS = {"normal": "#2E86AB", "failure": "#D1495B"}


@st.cache_data
def load_data():
    return add_status(load_raw_data())


apply_style()
hero(
    "Dataset Explorer",
    "Look at the AI4I 2020 data: filter it, compare normal and failed machines.",
)

try:
    data = load_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

left, right = st.columns(2)
with left:
    types = st.multiselect("Machine type", MACHINE_TYPES, default=MACHINE_TYPES)
with right:
    statuses = st.multiselect(
        "Status", list(STATUS_LABELS.values()), default=list(STATUS_LABELS.values())
    )

view = filter_data(data, types, statuses)
if view.empty:
    st.warning("No rows match these filters. Pick at least one type and one status.")
    st.stop()

cols = st.columns(3)
with cols[0]:
    metric_card("Rows shown", f"{len(view):,}", f"of {len(data):,} in the dataset")
with cols[1]:
    metric_card("Failures", f"{int(view[TARGET_COLUMN].sum()):,}", "in the rows shown")
with cols[2]:
    metric_card("Failure rate", f"{100 * view[TARGET_COLUMN].mean():.2f}%", "in the rows shown")

st.write("")
tab_table, tab_dist, tab_compare, tab_corr = st.tabs(
    ["Data table", "Distributions", "Normal vs failure", "Correlations"]
)

with tab_table:
    st.dataframe(view.head(500), width="stretch", hide_index=True)
    st.caption(
        "Showing the first 500 rows of the filter. UDI, Product ID, Machine failure and "
        f"{', '.join(FAILURE_MODE_COLUMNS)} are never given to the model as inputs."
    )

with tab_dist:
    sensor = st.selectbox("Sensor", NUMERIC_FEATURES, key="dist_sensor")
    figure = px.histogram(
        view, x=sensor, color="Status", nbins=40, barmode="overlay",
        opacity=0.7, color_discrete_map=COLORS,
    )
    st.plotly_chart(figure, width="stretch")
    st.caption("Outliers are kept: unusual readings are often the failures.")

with tab_compare:
    sensor = st.selectbox("Sensor", NUMERIC_FEATURES, key="compare_sensor")
    figure = px.box(
        view, x="Status", y=sensor, color="Status",
        category_orders={"Status": ["normal", "failure"]}, color_discrete_map=COLORS,
    )
    st.plotly_chart(figure, width="stretch")

with tab_corr:
    corr = view[NUMERIC_FEATURES + [TARGET_COLUMN]].corr().round(2)
    figure = px.imshow(
        corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "No single sensor strongly predicts failure on its own, which is why "
        "tree-based models work better than a straight-line model here."
    )

st.caption(
    "Data: AI4I 2020 Predictive Maintenance Dataset, S. Matzka (2020), UCI Machine "
    "Learning Repository, CC BY 4.0. The dataset is synthetic."
)
