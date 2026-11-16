"""Smoke tests for the Model Performance page."""

import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import PROJECT_ROOT

REPORT = PROJECT_ROOT / "reports" / "model_metrics.json"


@pytest.mark.skipif(not REPORT.is_file(), reason="run src.evaluate_model first")
def test_performance_page_runs_without_errors():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "performance.py"), default_timeout=30
    ).run()
    assert not app.exception


@pytest.mark.skipif(not REPORT.is_file(), reason="run src.evaluate_model first")
def test_performance_page_lists_all_four_models():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "performance.py"), default_timeout=30
    ).run()
    assert len(app.dataframe[0].value) == 4
