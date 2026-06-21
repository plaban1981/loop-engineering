# tests/test_tools.py
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
    assert result["flood_zone"] is True


def test_lookup_application_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown application"):
        lookup_application.invoke({"app_id": "APP-999"})


def test_retrieve_rules_flood():
    rules = retrieve_rules.invoke({"risk_factor": "flood"})
    assert isinstance(rules, list)
    assert any("flood_rider_required" in r for r in rules)


def test_retrieve_rules_no_match():
    rules = retrieve_rules.invoke({"risk_factor": "xyzzy_nonexistent"})
    assert rules == ["No specific rules found for this risk factor."]


def test_fetch_risk_signal_credit():
    result = fetch_risk_signal.invoke({"app_id": "APP-003", "signal_type": "credit"})
    assert result["tier"] == "poor"
    assert result["risk_label"] == "high"


def test_fetch_risk_signal_flood():
    result = fetch_risk_signal.invoke({"app_id": "APP-001", "signal_type": "flood"})
    assert result["flood_zone"] is True
    assert result["zone_code"] == "AE"


def test_fetch_risk_signal_claims():
    result = fetch_risk_signal.invoke({"app_id": "APP-002", "signal_type": "claims"})
    assert result["claim_count_5yr"] == 3


def test_fetch_risk_signal_property():
    result = fetch_risk_signal.invoke({"app_id": "APP-006", "signal_type": "property"})
    assert "estimated_value" in result
    assert "replacement_cost" in result


def test_fetch_risk_signal_unknown_type():
    with pytest.raises(ValueError, match="Unknown signal type"):
        fetch_risk_signal.invoke({"app_id": "APP-001", "signal_type": "mystery"})


def test_draft_decision_returns_structured():
    result = draft_decision.invoke({
        "decision": "decline",
        "rationale": "Excess claims history",
        "flags": ["excess_claims"],
        "recommended_premium_adj": 0.0,
    })
    assert result["decision"] == "decline"
    assert "excess_claims" in result["flags"]
