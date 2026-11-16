"""Tests for src/batch.py and the Batch Prediction page."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.batch import read_csv_bytes, risk_summary
from src.data_loader import PROJECT_ROOT
from src.predict import INFO_PATH, MODEL_PATH, InputValidationError


def test_reads_csv_with_a_bom_and_spaces_in_headers():
    data = "\ufeff Type ,Torque [Nm]\nM,40\n".encode("utf-8")
    table = read_csv_bytes(data)
    assert list(table.columns) == ["Type", "Torque [Nm]"]


def test_empty_file_is_rejected():
    with pytest.raises(InputValidationError, match="empty"):
        read_csv_bytes(b"")


def test_too_many_rows_are_rejected():
    with pytest.raises(InputValidationError, match="limit"):
        read_csv_bytes(b"a\n1\n2\n3\n", max_rows=2)


def test_risk_summary_always_lists_all_three_levels():
    result = pd.DataFrame({"risk_level": ["High", "Low", "Low"]})
    summary = risk_summary(result)
    assert list(summary.index) == ["High", "Medium", "Low"]
    assert summary["machines"].tolist() == [1, 0, 2]


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and INFO_PATH.is_file()),
    reason="final model not saved yet",
)
def test_batch_page_runs_without_errors():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "batch.py"), default_timeout=30
    ).run()
    assert not app.exception
