import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.application_tool import lookup_application
from src.tools.rules_tool import retrieve_rules
from src.tools.risk_signal_tool import fetch_risk_signal
from src.tools.decision_tool import draft_decision


def test_lookup_application_returns_record():
    result = lookup_application.invoke({"app_id": "APP-001"})
    assert result["id"] == "APP-001"
    assert result["state"] == "FL"


def test_lookup_application_raises_on_unknown():
    with pytest.raises(ValueError):
        lookup_application.invoke({"app_id": "APP-999"})


def test_retrieve_rules_flood():
    rules = retrieve_rules.invoke({"risk_factor": "flood"})
    assert any("flood_rider_required" in r for r in rules)


def test_fetch_risk_signal_credit():
    result = fetch_risk_signal.invoke({"app_id": "APP-003", "signal_type": "credit"})
    assert result["tier"] == "poor"


def test_draft_decision_structured():
    result = draft_decision.invoke({
        "decision": "decline",
        "rationale": "Poor credit",
        "flags": ["poor_credit"],
        "recommended_premium_adj": 0.0,
    })
    assert result["decision"] == "decline"
