import json
import time
from pathlib import Path
from typing import Callable

from .agent import build_verified_agent
from .audit import verify_judge_checksum
from .billing import CostMeter
from .judge import _extract_decision, is_correct
from .memory import distill_lesson, with_lessons

_STATE_DIR = Path(__file__).parent.parent / "state"
DEFAULT_PENDING = _STATE_DIR / "pending_applications.json"
DEFAULT_STATE = _STATE_DIR / "run_state.json"
HUMAN_QUEUE_PATH = _STATE_DIR / "human_queue.json"
HILL_CLIMB_EVERY = 8
POLL_INTERVAL = 10


def update_state(state: dict, app_id: str, passed: bool, lesson: str | None) -> dict:
    state = dict(state)
    state["processed"] = state.get("processed", []) + [app_id]
    state["pending"] = [x for x in state.get("pending", []) if x != app_id]
    key = "passed" if passed else "failed"
    state[key] = state.get(key, []) + [app_id]
    if lesson:
        state["lessons"] = state.get("lessons", []) + [lesson]
    state["decisions_since_last_improvement"] = state.get("decisions_since_last_improvement", 0) + 1
    return state


def run_hill_climbing(agent_factory: Callable, system_prompt: str, lessons: list, meter: CostMeter) -> str:
    from .hill_climbing import improve
    return improve(agent_factory, system_prompt, lessons, meter)


def run_event_loop(
    system_prompt: str,
    meter: CostMeter,
    pending_path: Path = DEFAULT_PENDING,
    state_path: Path = DEFAULT_STATE,
    once: bool = False,
) -> None:
    run_state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    verify_judge_checksum(run_state.get("judge_checksum", ""))

    total = len(json.loads(pending_path.read_text(encoding="utf-8-sig")) if pending_path.exists() else [])
    processed = 0

    while True:
        pending = json.loads(pending_path.read_text(encoding="utf-8-sig")) if pending_path.exists() else []

        if not pending:
            if once:
                print("[event_loop] queue empty — done")
                break
            print("[event_loop] queue empty — sleeping...")
            time.sleep(POLL_INTERVAL)
            continue

        app_id = pending[0]
        processed += 1
        print(f"[event_loop] {processed}/{total}  {app_id} ...", flush=True)
        lessons = run_state.get("lessons", [])
        prompt = with_lessons(system_prompt, lessons)

        from .tools.application_tool import lookup_application
        try:
            app_context = lookup_application.invoke({"app_id": app_id})
        except ValueError:
            app_context = {}

        agent = build_verified_agent(prompt, meter)
        result = agent.invoke({
            "input": f"Evaluate insurance application {app_id}. Use all tools and call draft_decision.",
            "_app_context": app_context,
        })

        decision = result.get("decision") or _extract_decision(result)

        if decision == "refer":
            queue = json.loads(HUMAN_QUEUE_PATH.read_text(encoding="utf-8-sig")) if HUMAN_QUEUE_PATH.exists() else []
            queue.append({"app_id": app_id, "result": result})
            HUMAN_QUEUE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")
            passed = True
        else:
            passed = is_correct(decision, app_id)

        status = "PASS" if passed else "FAIL"
        spend = f"${meter.spent:.4f}"
        print(f"[event_loop] {app_id}  decision={decision}  {status}  spend={spend}")

        lesson = None
        if not passed:
            lesson = distill_lesson(app_context, got=decision, expected="")
            print(f"[event_loop] {app_id}  lesson distilled")

        run_state = update_state(run_state, app_id, passed, lesson)
        state_path.write_text(json.dumps(run_state, indent=2), encoding="utf-8")

        remaining = [x for x in pending if x != app_id]
        pending_path.write_text(json.dumps(remaining), encoding="utf-8")

        if run_state["decisions_since_last_improvement"] >= HILL_CLIMB_EVERY:
            print(f"[event_loop] {HILL_CLIMB_EVERY} decisions reached — running hill-climbing...")
            system_prompt = run_hill_climbing(build_verified_agent, system_prompt, lessons, meter)
            run_state["decisions_since_last_improvement"] = 0
            state_path.write_text(json.dumps(run_state, indent=2), encoding="utf-8")
            print(f"[event_loop] hill-climbing done  spend={spend}")

        if once and not remaining:
            break
