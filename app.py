"""Streamlit entry point. Run with: streamlit run app.py"""

import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page("views/home.py", title="Home", icon="🏠", default=True),
        st.Page("views/prediction.py", title="Prediction", icon="🔮"),
        st.Page("views/batch.py", title="Batch Prediction", icon="📂"),
        st.Page("views/performance.py", title="Model Performance", icon="📊"),
        st.Page("views/explorer.py", title="Dataset Explorer", icon="🔎"),
        st.Page("views/insights.py", title="Maintenance Insights", icon="🧰"),
        st.Page("views/about.py", title="About", icon="ℹ️"),
    ]
)
navigation.run()
