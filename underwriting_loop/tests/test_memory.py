# tests/test_memory.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.lesson_memory import with_lessons, distill_lesson, measure_gain


def test_with_lessons_empty():
    base = "You are an underwriter."
    assert with_lessons(base, []) == base


def test_with_lessons_appends():
    base = "You are an underwriter."
    lessons = ["Flood zone without rider → modify.", "Claims > 2 → decline."]
    result = with_lessons(base, lessons)
    assert "Underwriting lessons" in result
    assert "Flood zone without rider" in result
    assert "Claims > 2" in result


def test_with_lessons_format():
    result = with_lessons("base", ["lesson one"])
    lines = result.split("\n")
    assert any(line.startswith("- lesson one") for line in lines)


def test_distill_lesson_calls_claude():
    fake_lesson = "Properties in flood zones without a rider must be modified."

    with patch("src.memory.lesson_memory.anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=fake_lesson)]
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        result = distill_lesson(
            application={"id": "APP-001", "flood_zone": True, "has_flood_rider": False},
            got="accept",
            expected="modify",
        )

    assert result == fake_lesson


def test_measure_gain_positive():
    with patch("src.memory.lesson_memory.evaluate_prompt") as mock_eval:
        mock_eval.side_effect = [(0.5, []), (0.75, [])]
        gain = measure_gain("base prompt", ["lesson 1"], split="test", meter=None)

    assert abs(gain - 0.25) < 0.001


def test_measure_gain_zero():
    with patch("src.memory.lesson_memory.evaluate_prompt") as mock_eval:
        mock_eval.side_effect = [(0.6, []), (0.6, [])]
        gain = measure_gain("base", [], split="test", meter=None)

    assert gain == 0.0
