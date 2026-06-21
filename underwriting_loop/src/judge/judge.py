# src/judge/judge.py
import json
from pathlib import Path

_HIST = Path(__file__).parent.parent / "data" / "historical_decisions.json"
_APPS = Path(__file__).parent.parent / "data" / "applications.json"


def load_historical_decisions() -> dict:
    return json.loads(_HIST.read_text())


def load_tasks(split: str) -> list[dict]:
    decisions = load_historical_decisions()
    apps = {a["id"]: a for a in json.loads(_APPS.read_text())}
    return [
        {
            "id": app_id,
            "split": meta["split"],
            "expected_decision": meta["expected_decision"],
            "expected_flags": meta["expected_flags"],
            "application": apps[app_id],
        }
        for app_id, meta in decisions.items()
        if meta["split"] == split
    ]


def is_correct(agent_decision: str, app_id: str) -> bool:
    decisions = load_historical_decisions()
    if app_id not in decisions:
        return True  # no ground truth for this ID (unseen production app)
    return agent_decision == decisions[app_id]["expected_decision"]


def evaluate_prompt(
    system_prompt: str,
    split: str = "train",
    meter=None,
) -> tuple[float, list[dict]]:
    """Run every task in split through Loop 1. Returns (accuracy, failures). Never calls LLM internally."""
    from ..loops.loop1_agent import run_loop1  # deferred to avoid circular import at module load

    tasks = load_tasks(split)
    failures = []
    for task in tasks:
        state = run_loop1(task["id"], system_prompt, meter)
        decision = state.get("decision", "ran_out_of_steps")
        if not is_correct(decision, task["id"]):
            failures.append({
                "app_id": task["id"],
                "got": decision,
                "expected": task["expected_decision"],
                "application": task["application"],
            })
    score = (len(tasks) - len(failures)) / len(tasks)
    return score, failures
