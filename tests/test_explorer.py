"""Tests for src/explorer.py and the Dataset Explorer page."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import PROJECT_ROOT, RAW_DATA_PATH
from src.explorer import add_status, filter_data


def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Type": ["L", "L", "M", "H"], "Machine failure": [0, 1, 0, 1]}
    )


def test_status_labels_are_added_without_changing_the_input():
    original = tiny_df()
    table = add_status(original)
    assert table["Status"].tolist() == ["normal", "failure", "normal", "failure"]
    assert "Status" not in original.columns


def test_filter_by_type_and_status():
    table = add_status(tiny_df())
    result = filter_data(table, ["L", "H"], ["failure"])
    assert result["Type"].tolist() == ["L", "H"]
    assert filter_data(table, [], ["normal", "failure"]).empty


@pytest.mark.skipif(not RAW_DATA_PATH.is_file(), reason="dataset not uploaded")
def test_explorer_page_runs_without_errors():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "explorer.py"), default_timeout=60
    ).run()
    assert not app.exception
