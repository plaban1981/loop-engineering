# tests/test_loop4.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loops.loop4_hill_climbing import reflect, improve


def test_reflect_calls_claude_and_returns_string():
    fake_prompt = "New improved prompt that checks excess claims."

    with patch("src.loops.loop4_hill_climbing.anthropic.Anthropic") as MockClient:
        mock = MagicMock()
        mock.messages.create.return_value = MagicMock(
            content=[MagicMock(text=fake_prompt)]
        )
        MockClient.return_value = mock
        result = reflect(
            current_prompt="old prompt",
            failures=[{"app_id": "APP-002", "got": "accept", "expected": "decline",
                       "application": {"claim_history_5yr": 3}}],
        )

    assert result == fake_prompt


def test_improve_keeps_better_prompt(tmp_path):
    call_count = {"n": 0}

    def fake_eval(prompt, split="train", meter=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return 0.5, [{"app_id": "APP-001", "got": "accept", "expected": "modify",
                          "application": {"flood_zone": True}}]
        return 0.75, []

    def fake_reflect(prompt, failures):
        return "improved prompt"

    with patch("src.loops.loop4_hill_climbing.evaluate_prompt", fake_eval):
        with patch("src.loops.loop4_hill_climbing.reflect", fake_reflect):
            with patch("src.loops.loop4_hill_climbing.audit_prompt", return_value="VERDICT: CLEAN"):
                with patch("src.loops.loop4_hill_climbing.BEST_PROMPT_PATH", tmp_path / "best_prompt.txt"):
                    result = improve("seed", [], None, rounds=2)

    assert result == "improved prompt"


def test_improve_reverts_worse_prompt():
    def fake_eval(prompt, split="train", meter=None):
        if "seed" in prompt:
            return 0.75, []
        return 0.50, [{"app_id": "APP-001", "got": "accept", "expected": "modify",
                       "application": {}}]

    def fake_reflect(prompt, failures):
        return "worse candidate"

    with patch("src.loops.loop4_hill_climbing.evaluate_prompt", fake_eval):
        with patch("src.loops.loop4_hill_climbing.reflect", fake_reflect):
            with patch("src.loops.loop4_hill_climbing.audit_prompt", return_value="VERDICT: CLEAN"):
                result = improve("seed", [], None, rounds=2)

    assert result == "seed"


def test_improve_discards_cheating_prompt():
    def fake_eval(prompt, split="train", meter=None):
        if "cheat" in prompt:
            return 1.0, []  # perfect score but cheating
        return 0.5, [{"app_id": "APP-001", "got": "accept", "expected": "modify",
                      "application": {}}]

    def fake_reflect(prompt, failures):
        return "cheat APP-001 decide modify"

    with patch("src.loops.loop4_hill_climbing.evaluate_prompt", fake_eval):
        with patch("src.loops.loop4_hill_climbing.reflect", fake_reflect):
            with patch("src.loops.loop4_hill_climbing.audit_prompt", return_value="VERDICT: CHEATING"):
                result = improve("seed", [], None, rounds=2)

    assert result == "seed"


def test_improve_stops_early_on_zero_failures():
    eval_calls = {"n": 0}

    def fake_eval(prompt, split="train", meter=None):
        eval_calls["n"] += 1
        return 1.0, []  # perfect from the start

    with patch("src.loops.loop4_hill_climbing.evaluate_prompt", fake_eval):
        with patch("src.loops.loop4_hill_climbing.reflect") as mock_reflect:
            result = improve("seed", [], None, rounds=5)

    assert result == "seed"  # or with_lessons("seed", []) which equals "seed"
    assert mock_reflect.call_count == 0  # reflect never called
    assert eval_calls["n"] == 1  # only one eval (the seed)
