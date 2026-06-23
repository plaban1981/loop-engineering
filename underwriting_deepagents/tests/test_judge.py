import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.judge import is_correct, load_tasks, evaluate
from src.audit import compute_judge_checksum, verify_judge_checksum
from src.billing import CostMeter, BudgetExhaustedError


# --- billing ---

def test_cost_meter_records():
    m = CostMeter()
    m.record(1000, 100)
    assert m.spent > 0


def test_cost_meter_budget():
    m = CostMeter(budget_usd=0.001)
    m.record(100_000, 0)
    with pytest.raises(BudgetExhaustedError):
        m.record(1, 0)


# --- audit ---

def test_checksum_deterministic():
    c1 = compute_judge_checksum()
    c2 = compute_judge_checksum()
    assert c1 == c2 and len(c1) == 64


def test_checksum_verify_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_judge_checksum("bad" * 20)


# --- judge ---

def test_is_correct_modify():
    assert is_correct("modify", "APP-001") is True


def test_is_correct_wrong():
    assert is_correct("accept", "APP-001") is False


def test_load_tasks_split():
    train = load_tasks("train")
    test = load_tasks("test")
    assert len(train) == 16
    assert len(test) == 4


def test_evaluate_calls_agent_factory():
    def fake_factory(prompt, meter):
        agent = MagicMock()
        agent.invoke.return_value = {"output": "decision: accept"}
        return agent

    with patch("src.judge._extract_decision", return_value="accept"):
        score, failures = evaluate(fake_factory, "seed prompt", split="train", meter=None)

    # train accept apps: 006,008,011,013,015 = 5 out of 16
    assert abs(score - 5 / 16) < 0.01
    assert len(failures) == 11
