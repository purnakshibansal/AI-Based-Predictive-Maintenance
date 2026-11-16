"""Tests for src/recommendations.py and the Maintenance Insights page."""

import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import PROJECT_ROOT, RAW_DATA_PATH
from src.predict import INFO_PATH, MODEL_PATH
from src.recommendations import check_failure_rules

TYPICAL = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
}


def modes(**changes) -> set[str]:
    return {rule["mode"] for rule in check_failure_rules({**TYPICAL, **changes})}


def test_typical_machine_triggers_no_rule():
    assert modes() == set()


def test_heat_dissipation_rule():
    assert modes(**{"Process temperature [K]": 306.0, "Rotational speed [rpm]": 1300}) >= {"HDF"}
    assert "HDF" not in modes(**{"Rotational speed [rpm]": 1300})  # temperature gap is 10.5 K


def test_power_rule_high_and_low():
    assert "PWF" in modes(**{"Torque [Nm]": 70.0, "Rotational speed [rpm]": 1500})
    assert "PWF" in modes(**{"Torque [Nm]": 10.0, "Rotational speed [rpm]": 1500})


def test_overstrain_limit_depends_on_machine_type():
    heavy = {"Tool wear [min]": 200, "Torque [Nm]": 60.0}  # 12,000 minNm
    assert "OSF" in modes(Type="L", **heavy)
    assert "OSF" not in modes(Type="M", **heavy)  # exactly at the M limit, not above


def test_tool_wear_rule():
    assert "TWF" in modes(**{"Tool wear [min]": 210})
    assert "TWF" not in modes(**{"Tool wear [min]": 100})


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and INFO_PATH.is_file() and RAW_DATA_PATH.is_file()),
    reason="model or dataset missing",
)
def test_insights_page_runs_and_checks_a_machine():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "views" / "insights.py"), default_timeout=60
    ).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error) + len(app.warning) + len(app.success) >= 1
