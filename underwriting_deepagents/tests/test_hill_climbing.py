import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hill_climbing import reflect, improve
from src.agent import build_verified_agent


def test_improve_keeps_better():
    call_count = {"n": 0}

    def fake_eval(factory, prompt, split="train", meter=None):
        call_count["n"] += 1
        if call_count["n"] <= 1:
            return 0.5, [{"app_id": "APP-002", "got": "accept", "expected": "decline",
                          "application": {"claim_history_5yr": 3}}]
        return 0.75, []

    def fake_reflect(prompt, failures):
        return "improved prompt"

    with patch("src.hill_climbing.evaluate", fake_eval):
        with patch("src.hill_climbing.reflect", fake_reflect):
            with patch("src.hill_climbing.audit_prompt", return_value="VERDICT: CLEAN"):
                result = improve(build_verified_agent, "seed", [], None, rounds=2)

    assert result == "improved prompt"


def test_improve_reverts_cheating():
    def fake_eval(factory, prompt, split="train", meter=None):
        if "APP-001" in prompt:
            return 1.0, []
        return 0.5, [{"app_id": "APP-001", "got": "accept", "expected": "modify",
                      "application": {}}]

    def fake_reflect(prompt, failures):
        return "If APP-001 then modify."

    with patch("src.hill_climbing.evaluate", fake_eval):
        with patch("src.hill_climbing.reflect", fake_reflect):
            with patch("src.hill_climbing.audit_prompt", return_value="VERDICT: CHEATING"):
                result = improve(build_verified_agent, "seed", [], None, rounds=2)

    assert result == "seed"
