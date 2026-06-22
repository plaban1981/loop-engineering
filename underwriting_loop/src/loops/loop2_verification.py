# src/loops/loop2_verification.py
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from ..billing.cost_meter import CostMeter
from ..judge.judge import is_correct, load_historical_decisions
from .loop1_agent import run_loop1

HUMAN_QUEUE_PATH = Path(__file__).parent.parent.parent / "state" / "human_queue.json"


def _write_human_queue(app_id: str, state: dict) -> None:
    queue = []
    if HUMAN_QUEUE_PATH.exists():
        queue = json.loads(HUMAN_QUEUE_PATH.read_text())
    queue.append({
        "app_id": app_id,
        "decision": state.get("decision"),
        "rationale": state.get("rationale"),
    })
    HUMAN_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def run_loop2(
    app_id: str,
    system_prompt: str,
    meter: CostMeter,
    lessons: list[str] = None,
    max_retries: int = 3,
) -> tuple[dict, bool]:
    """Run agent with grader-based retry. Returns (final_state, passed)."""
    extra_messages = []
    state = run_loop1(app_id, system_prompt, meter, lessons)

    for attempt in range(max_retries):
        decision = state.get("decision", "ran_out_of_steps")

        if decision == "refer":
            _write_human_queue(app_id, state)
            return state, True

        if is_correct(decision, app_id):
            return state, True

        if attempt >= max_retries - 1:
            return state, False

        decisions = load_historical_decisions()
        expected = decisions.get(app_id, {}).get("expected_decision", "unknown")
        feedback = HumanMessage(
            content=(
                f"Your decision was '{decision}' but this application requires '{expected}'. "
                f"Review the risk factors carefully and call draft_decision again with the correct decision."
            )
        )
        extra_messages = list(state.get("messages", [])) + [feedback]
        state = run_loop1(app_id, system_prompt, meter, lessons, extra_messages)

    return state, False