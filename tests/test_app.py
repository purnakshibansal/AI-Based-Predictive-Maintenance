"""Smoke test: the dashboard starts without errors."""

import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import PROJECT_ROOT
from src.predict import INFO_PATH, MODEL_PATH


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and INFO_PATH.is_file()),
    reason="final model not saved yet",
)
def test_home_page_runs_without_errors():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and INFO_PATH.is_file()),
    reason="final model not saved yet",
)
def test_prediction_page_gives_a_risk_message_after_click():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "prediction.py"), default_timeout=30
    ).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error) + len(app.warning) + len(app.success) >= 1


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and INFO_PATH.is_file()),
    reason="final model not saved yet",
)
def test_about_page_runs_without_errors():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "about.py"), default_timeout=30
    ).run()
    assert not app.exception
