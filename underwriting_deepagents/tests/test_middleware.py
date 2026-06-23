import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.middleware import RubricMiddleware, check_rubric, UNDERWRITING_RUBRIC


def test_check_rubric_accept_valid():
    output = {
        "decision": "accept",
        "rationale": "No risk factors identified.",
        "flags": [],
        "recommended_premium_adj": 0.0,
        "app": {"flood_zone": False, "claim_history_5yr": 1, "credit_score_tier": "good",
                "coverage_amount": 500000, "state": "CA", "has_flood_rider": False},
    }
    violations = check_rubric(output, UNDERWRITING_RUBRIC)
    assert violations == []


def test_check_rubric_detects_missing_flood_flag():
    output = {
        "decision": "modify",
        "rationale": "Flood zone property.",
        "flags": [],  # missing flood_rider_required
        "recommended_premium_adj": 0.0,
        "app": {"flood_zone": True, "claim_history_5yr": 0, "credit_score_tier": "good",
                "coverage_amount": 300000, "state": "OH", "has_flood_rider": False},
    }
    violations = check_rubric(output, UNDERWRITING_RUBRIC)
    assert any("flood_rider_required" in v for v in violations)


def test_check_rubric_detects_wrong_decision_for_excess_claims():
    output = {
        "decision": "accept",  # wrong — should be decline or refer
        "rationale": "Risk acceptable.",
        "flags": [],
        "recommended_premium_adj": 0.0,
        "app": {"flood_zone": False, "claim_history_5yr": 3, "credit_score_tier": "good",
                "coverage_amount": 400000, "state": "CA", "has_flood_rider": False},
    }
    violations = check_rubric(output, UNDERWRITING_RUBRIC)
    assert any("excess_claims" in v or "claim_history" in v.lower() for v in violations)


def test_rubric_middleware_retries_on_violation():
    call_count = {"n": 0}
    mock_agent = MagicMock()

    def fake_invoke(inputs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "output": '{"decision": "accept", "rationale": "ok", "flags": [], "recommended_premium_adj": 0.0}',
                "_app": {"flood_zone": True, "claim_history_5yr": 0, "credit_score_tier": "good",
                         "coverage_amount": 300000, "state": "CA", "has_flood_rider": False},
            }
        return {
            "output": '{"decision": "modify", "rationale": "flood zone", "flags": ["flood_rider_required"], "recommended_premium_adj": 0.0}',
            "_app": {"flood_zone": True, "claim_history_5yr": 0, "credit_score_tier": "good",
                     "coverage_amount": 300000, "state": "CA", "has_flood_rider": False},
        }

    mock_agent.invoke = fake_invoke
    wrapped = RubricMiddleware(mock_agent, rubric=UNDERWRITING_RUBRIC, max_retries=3)
    result = wrapped.invoke({
        "input": "Evaluate APP-007",
        "_app_context": {"flood_zone": True, "claim_history_5yr": 0, "credit_score_tier": "good",
                         "coverage_amount": 300000, "state": "CA", "has_flood_rider": False},
    })

    assert call_count["n"] == 2
