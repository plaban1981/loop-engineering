# tests/test_judge.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.judge.judge import is_correct, load_historical_decisions, load_tasks
from src.judge.audit import compute_judge_checksum, verify_judge_checksum


def test_is_correct_modify():
    assert is_correct("modify", "APP-001") is True


def test_is_correct_false():
    assert is_correct("accept", "APP-001") is False


def test_is_correct_refer():
    assert is_correct("refer", "APP-004") is True


def test_is_correct_unknown_app_returns_true():
    # No ground truth available → treat as pass (production app)
    assert is_correct("accept", "APP-999") is True


def test_load_tasks_train_only():
    tasks = load_tasks("train")
    assert all(t["split"] == "train" for t in tasks)
    assert len(tasks) == 16


def test_load_tasks_test_only():
    tasks = load_tasks("test")
    assert all(t["split"] == "test" for t in tasks)
    assert len(tasks) == 4


def test_load_tasks_includes_application_data():
    tasks = load_tasks("train")
    first = tasks[0]
    assert "application" in first
    assert "id" in first["application"]


def test_judge_checksum_is_deterministic():
    c1 = compute_judge_checksum()
    c2 = compute_judge_checksum()
    assert c1 == c2
    assert len(c1) == 64  # SHA-256 hex


def test_verify_judge_checksum_passes():
    good = compute_judge_checksum()
    verify_judge_checksum(good)  # should not raise


def test_verify_judge_checksum_fails():
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_judge_checksum("deadbeef" * 8)
