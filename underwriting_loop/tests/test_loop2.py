# tests/test_loop2.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loops.loop2_verification import run_loop2
from src.billing.cost_meter import CostMeter


def make_meter():
    m = MagicMock(spec=CostMeter)
    m.spent = 0.0
    m.record_from_message = MagicMock()
    return m


def test_loop2_passes_on_correct_decision():
    def fake_loop1(app_id, prompt, meter, lessons=None, extra_messages=None):
        return {"decision": "modify", "rationale": "flood zone", "flags": [], "recommended_premium_adj": 0.0}

    with patch("src.loops.loop2_verification.run_loop1", fake_loop1):
        state, passed = run_loop2("APP-001", "seed", make_meter())

    assert passed is True
    assert state["decision"] == "modify"


def test_loop2_retries_on_wrong_decision():
    call_count = {"n": 0}

    def fake_loop1(app_id, prompt, meter, lessons=None, extra_messages=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"decision": "accept", "rationale": "all good", "flags": [], "recommended_premium_adj": 0.0}
        return {"decision": "modify", "rationale": "fixed", "flags": [], "recommended_premium_adj": 0.0}

    with patch("src.loops.loop2_verification.run_loop1", fake_loop1):
        state, passed = run_loop2("APP-001", "seed", make_meter())

    assert call_count["n"] == 2
    assert passed is True


def test_loop2_exhausts_retries():
    def fake_loop1(app_id, prompt, meter, lessons=None, extra_messages=None):
        return {"decision": "accept", "rationale": "wrong", "flags": [], "recommended_premium_adj": 0.0}

    with patch("src.loops.loop2_verification.run_loop1", fake_loop1):
        state, passed = run_loop2("APP-001", "seed", make_meter(), max_retries=3)

    assert passed is False
    assert state["decision"] == "accept"


def test_loop2_refer_skips_grader(tmp_path):
    def fake_loop1(app_id, prompt, meter, lessons=None, extra_messages=None):
        return {"decision": "refer", "rationale": "senior review", "flags": [], "recommended_premium_adj": 0.0}

    human_queue = tmp_path / "human_queue.json"
    with patch("src.loops.loop2_verification.run_loop1", fake_loop1):
        with patch("src.loops.loop2_verification.HUMAN_QUEUE_PATH", human_queue):
            state, passed = run_loop2("APP-004", "seed", make_meter())

    assert passed is True
    assert state["decision"] == "refer"
    assert human_queue.exists()
    queue = json.loads(human_queue.read_text())
    assert any(item["app_id"] == "APP-004" for item in queue)
