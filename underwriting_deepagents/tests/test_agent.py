import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import build_agent, build_verified_agent, SEED_PROMPT
from src.billing import CostMeter


def test_seed_prompt_exists():
    assert isinstance(SEED_PROMPT, str)
    assert len(SEED_PROMPT) > 50


def test_build_agent_returns_runnable():
    with patch("src.agent.ChatAnthropic"):
        with patch("src.agent.create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent = build_agent("test prompt", MagicMock(spec=CostMeter))
    assert agent is not None


def test_build_verified_agent_wraps_with_middleware():
    with patch("src.agent.ChatAnthropic"):
        with patch("src.agent.create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent = build_verified_agent("test prompt", MagicMock(spec=CostMeter))
    from src.middleware import RubricMiddleware
    assert isinstance(agent, RubricMiddleware)
