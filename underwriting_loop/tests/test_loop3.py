# tests/test_loop3.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loops.loop3_event import run_event_loop, load_run_state, save_run_state, update_state


def test_load_run_state(tmp_path):
    state_file = tmp_path / "run_state.json"
    state_file.write_text(json.dumps({
        "processed": [], "passed": [], "failed": [], "pending": ["APP-001"],
        "lessons": [], "decisions_since_last_improvement": 0, "judge_checksum": ""
    }))
    state = load_run_state(state_file)
    assert state["pending"] == ["APP-001"]


def test_update_state_on_pass():
    initial = {"processed": [], "passed": [], "failed": [], "pending": ["APP-001"],
               "lessons": [], "decisions_since_last_improvement": 0, "judge_checksum": ""}
    state = update_state(initial, "APP-001", passed=True, lesson=None)
    assert "APP-001" in state["processed"]
    assert "APP-001" in state["passed"]
    assert state["decisions_since_last_improvement"] == 1
    assert "APP-001" not in state["pending"]


def test_update_state_on_fail_with_lesson():
    initial = {"processed": [], "passed": [], "failed": [], "pending": ["APP-002"],
               "lessons": [], "decisions_since_last_improvement": 0, "judge_checksum": ""}
    state = update_state(initial, "APP-002", passed=False, lesson="Check excess claims.")
    assert "APP-002" in state["failed"]
    assert "Check excess claims." in state["lessons"]


def test_update_state_increments_counter():
    initial = {"processed": [], "passed": [], "failed": [], "pending": ["APP-003"],
               "lessons": [], "decisions_since_last_improvement": 5, "judge_checksum": ""}
    state = update_state(initial, "APP-003", passed=True, lesson=None)
    assert state["decisions_since_last_improvement"] == 6


def test_run_event_loop_once(tmp_path):
    pending_file = tmp_path / "pending.json"
    state_file = tmp_path / "run_state.json"
    pending_file.write_text(json.dumps(["APP-001"]))
    state_file.write_text(json.dumps({
        "processed": [], "passed": [], "failed": [], "pending": ["APP-001"],
        "lessons": [], "decisions_since_last_improvement": 0, "judge_checksum": "abc"
    }))

    def fake_loop2(app_id, prompt, meter, lessons=None, max_retries=3):
        return ({"decision": "modify", "rationale": "ok", "flags": [], "recommended_premium_adj": 0.0}, True)

    with patch("src.loops.loop3_event.run_loop2", fake_loop2):
        with patch("src.loops.loop3_event.verify_judge_checksum"):
            with patch("src.loops.loop3_event.run_hill_climbing", return_value="seed"):
                run_event_loop(
                    system_prompt="seed",
                    meter=MagicMock(),
                    pending_path=pending_file,
                    state_path=state_file,
                    once=True,
                )

    final = json.loads(state_file.read_text())
    assert "APP-001" in final["processed"]
    assert "APP-001" in final["passed"]
