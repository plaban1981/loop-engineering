import json
import re
from pathlib import Path
from typing import Callable

_HIST = Path(__file__).parent / "data" / "historical_decisions.json"
_APPS = Path(__file__).parent / "data" / "applications.json"


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
            "application": apps[app_id],
        }
        for app_id, meta in decisions.items()
        if meta["split"] == split
    ]


def is_correct(agent_decision: str, app_id: str) -> bool:
    decisions = load_historical_decisions()
    if app_id not in decisions:
        return True
    return agent_decision == decisions[app_id]["expected_decision"]


def _extract_decision(result: dict | str) -> str:
    """Extract decision string from agent output."""
    if isinstance(result, dict):
        # RubricMiddleware returns a parsed dict with decision directly
        decision = result.get("decision", "")
        if decision in {"accept", "modify", "decline", "refer"}:
            return decision
        text = result.get("output", str(result))
    else:
        text = str(result)
    for word in ["accept", "modify", "decline", "refer"]:
        if f"decision: {word}" in text.lower() or f'"decision": "{word}"' in text.lower():
            return word
    for word in ["decline", "refer", "modify", "accept"]:
        if word in text.lower():
            return word
    return "ran_out_of_steps"


def evaluate(
    agent_factory: Callable,
    system_prompt: str,
    split: str = "train",
    meter=None,
) -> tuple[float, list[dict]]:
    """Evaluate system_prompt on all tasks in split. agent_factory(prompt, meter) → agent Runnable."""
    tasks = load_tasks(split)
    failures = []
    for task in tasks:
        agent = agent_factory(system_prompt, meter)
        result = agent.invoke({"input": f"Evaluate insurance application {task['id']}. Use all available tools and call draft_decision with your final decision."})
        decision = _extract_decision(result)
        if not is_correct(decision, task["id"]):
            failures.append({
                "app_id": task["id"],
                "got": decision,
                "expected": task["expected_decision"],
                "application": task["application"],
            })
    score = (len(tasks) - len(failures)) / len(tasks)
    return score, failures
