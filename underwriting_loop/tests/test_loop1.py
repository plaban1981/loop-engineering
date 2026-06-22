# tests/test_loop1.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage

from src.loops.loop1_agent import run_loop1, SEED_PROMPT, UnderwritingState
from src.billing.cost_meter import CostMeter


def make_mock_meter():
    m = MagicMock(spec=CostMeter)
    m.spent = 0.0
    m.record_from_message = MagicMock()
    return m


def test_seed_prompt_exists():
    assert isinstance(SEED_PROMPT, str)
    assert len(SEED_PROMPT) > 50
    assert "draft_decision" in SEED_PROMPT


def test_underwriting_state_has_required_keys():
    # TypedDict keys can be inspected via __annotations__
    annotations = UnderwritingState.__annotations__
    for key in ["application_id", "messages", "decision", "rationale", "flags",
                "recommended_premium_adj", "lessons", "spend"]:
        assert key in annotations, f"Missing key: {key}"


def test_run_loop1_returns_dict_with_decision(monkeypatch):
    """run_loop1 returns a dict with at minimum a 'decision' key."""
    # Use a real AIMessage so LangGraph's add_messages reducer doesn't choke
    # usage_metadata requires total_tokens field
    mock_response = AIMessage(
        content="I recommend declining this application.",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )

    with patch("src.loops.loop1_agent.ChatAnthropic") as MockChat:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = mock_response
        MockChat.return_value = mock_model

        state = run_loop1("APP-002", SEED_PROMPT, make_mock_meter())

    assert isinstance(state, dict)
    assert "decision" in state


def test_run_loop1_out_of_steps_sets_decision(monkeypatch):
    """If agent never calls draft_decision, decision is 'ran_out_of_steps'."""
    tool_call = {"name": "lookup_application", "args": {"app_id": "APP-001"}, "id": "tc1"}
    # Use a real AIMessage with a tool_calls list
    mock_response = AIMessage(
        content="",
        tool_calls=[tool_call],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    with patch("src.loops.loop1_agent.ChatAnthropic") as MockChat:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = mock_response
        MockChat.return_value = mock_model

        # Patch _dispatch_tools to return empty results so the graph doesn't break
        with patch("src.loops.loop1_agent._dispatch_tools", return_value=([], {})):
            state = run_loop1("APP-001", SEED_PROMPT, make_mock_meter())

    assert state["decision"] == "ran_out_of_steps"
