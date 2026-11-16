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
    ]
)
navigation.run()
