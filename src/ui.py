"""Shared look and helpers for the Streamlit dashboard."""

import json

import streamlit as st

from src.predict import INFO_PATH

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {border-left: 6px solid #F59E0B; padding: 0.4rem 1rem; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: 0.3rem 0 0 0; color: #9FB0C0;}
.card {background: #1A222C; border: 1px solid #2A3644; border-radius: 10px;
       padding: 0.9rem 1.1rem; height: 100%;}
.card .label {color: #9FB0C0; font-size: 0.8rem; text-transform: uppercase;
              letter-spacing: 0.06em;}
.card .value {font-size: 1.9rem; font-weight: 700; color: #F59E0B;}
.card .note {color: #9FB0C0; font-size: 0.8rem;}
</style>
"""


def apply_style() -> None:
    """Inject the shared CSS. Call once per page."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    """Page title block."""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    """One dark metric card."""
    st.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_model_info() -> dict:
    """Read models/model_info.json (written by src.finalize_model)."""
    if not INFO_PATH.is_file():
        raise FileNotFoundError(
            "models/model_info.json not found. Run `python -m src.finalize_model`."
        )
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_cached_model():
    """Load the saved model once and reuse it between page runs."""
    from src.predict import load_model

    return load_model()
