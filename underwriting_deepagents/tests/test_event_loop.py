import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.event_loop import run_event_loop, update_state


def test_update_state_pass():
    state = {"processed": [], "passed": [], "failed": [], "pending": ["APP-001"],
             "lessons": [], "decisions_since_last_improvement": 0}
    updated = update_state(state, "APP-001", passed=True, lesson=None)
    assert "APP-001" in updated["processed"]
    assert "APP-001" in updated["passed"]
    assert updated["decisions_since_last_improvement"] == 1


def test_run_event_loop_processes_one(tmp_path):
    pending_file = tmp_path / "pending.json"
    state_file = tmp_path / "run_state.json"
    pending_file.write_text(json.dumps(["APP-001"]))
    state_file.write_text(json.dumps({
        "processed": [], "passed": [], "failed": [], "pending": ["APP-001"],
        "lessons": [], "decisions_since_last_improvement": 0, "judge_checksum": "x"
    }))

    def fake_verified_agent_invoke(inputs):
        return {"decision": "modify", "rationale": "flood", "flags": [], "recommended_premium_adj": 0.0}

    fake_agent = MagicMock()
    fake_agent.invoke = fake_verified_agent_invoke

    with patch("src.event_loop.build_verified_agent", return_value=fake_agent):
        with patch("src.event_loop.verify_judge_checksum"):
            with patch("src.event_loop.run_hill_climbing", return_value="seed"):
                with patch("src.event_loop._extract_decision", return_value="modify"):
                    run_event_loop(
                        system_prompt="seed",
                        meter=MagicMock(),
                        pending_path=pending_file,
                        state_path=state_file,
                        once=True,
                    )

    final = json.loads(state_file.read_text())
    assert "APP-001" in final["processed"]
