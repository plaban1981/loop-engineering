# tests/test_cost_meter.py
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.billing.cost_meter import CostMeter, BudgetExhaustedError


def test_record_adds_cost():
    meter = CostMeter()
    meter.record(input_tokens=1000, output_tokens=100)
    # 1000 * 3/1_000_000 + 100 * 15/1_000_000 = 0.003 + 0.0015 = 0.0045
    assert abs(meter.spent - 0.0045) < 1e-6


def test_record_accumulates():
    meter = CostMeter()
    meter.record(1000, 100)
    meter.record(1000, 100)
    assert abs(meter.spent - 0.009) < 1e-6


def test_budget_exceeded_raises():
    meter = CostMeter(budget_usd=0.001)
    meter.record(100_000, 0)  # 100k input tokens = $0.30 >> $0.001
    with pytest.raises(BudgetExhaustedError):
        meter.record(1, 0)


def test_record_from_message_noop_when_no_metadata():
    meter = CostMeter()

    class FakeMsg:
        usage_metadata = None

    meter.record_from_message(FakeMsg())
    assert meter.spent == 0.0


def test_record_from_message_uses_metadata():
    meter = CostMeter()

    class FakeMsg:
        usage_metadata = {"input_tokens": 500, "output_tokens": 50}

    meter.record_from_message(FakeMsg())
    assert meter.spent > 0


def test_initial_spent_is_zero():
    meter = CostMeter()
    assert meter.spent == 0.0
