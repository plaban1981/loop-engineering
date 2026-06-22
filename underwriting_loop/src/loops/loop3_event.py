# src/loops/loop3_event.py
import json
import time
from pathlib import Path

from ..billing.cost_meter import BudgetExhaustedError, CostMeter
from ..judge.audit import verify_judge_checksum
from ..memory.lesson_memory import distill_lesson, with_lessons
from .loop2_verification import run_loop2

_STATE_DIR = Path(__file__).parent.parent.parent / "state"
DEFAULT_PENDING = _STATE_DIR / "pending_applications.json"
DEFAULT_STATE = _STATE_DIR / "run_state.json"
HILL_CLIMB_EVERY = 8
POLL_INTERVAL = 10


def load_run_state(path: Path) -> dict:
    return json.loads(path.read_text())


def save_run_state(state: dict, path: Path) -> None:
    path.write_text(json.dumps(state, indent=2))


def update_state(state: dict, app_id: str, passed: bool, lesson: str | None) -> dict:
    state = dict(state)
    state["processed"] = state.get("processed", []) + [app_id]
    state["pending"] = [x for x in state.get("pending", []) if x != app_id]
    if passed:
        state["passed"] = state.get("passed", []) + [app_id]
    else:
        state["failed"] = state.get("failed", []) + [app_id]
        if lesson:
            state["lessons"] = state.get("lessons", []) + [lesson]
    state["decisions_since_last_improvement"] = state.get("decisions_since_last_improvement", 0) + 1
    return state


def run_hill_climbing(system_prompt: str, lessons: list[str], meter: CostMeter) -> str:
    from .loop4_hill_climbing import improve  # deferred to avoid circular import
    return improve(system_prompt, lessons, meter)


def run_event_loop(
    system_prompt: str,
    meter: CostMeter,
    pending_path: Path = DEFAULT_PENDING,
    state_path: Path = DEFAULT_STATE,
    once: bool = False,
) -> None:
    run_state = load_run_state(state_path)
    verify_judge_checksum(run_state.get("judge_checksum", ""))

    while True:
        pending = json.loads(pending_path.read_text()) if pending_path.exists() else []

        if not pending:
            if once:
                break
            print("Queue empty. Sleeping...")
            time.sleep(POLL_INTERVAL)
            continue

        app_id = pending[0]
        lessons = run_state.get("lessons", [])
        prompt_with_lessons = with_lessons(system_prompt, lessons)

        try:
            result_state, passed = run_loop2(app_id, prompt_with_lessons, meter)
        except BudgetExhaustedError:
            raise  # budget exhaustion should propagate — do not swallow
        except Exception as exc:
            print(f"[loop3] {app_id} errored: {exc!r} — skipping")
            remaining = [x for x in pending if x != app_id]
            pending_path.write_text(json.dumps(remaining))
            continue

        lesson = None
        if not passed:
            from ..judge.judge import load_historical_decisions
            from ..tools.application_tool import lookup_application
            try:
                app_data = lookup_application.invoke({"app_id": app_id})
            except Exception:
                app_data = {}
            historical = load_historical_decisions()
            expected_decision = historical.get(app_id, {}).get("expected_decision", "")
            lesson = distill_lesson(
                application=app_data,
                got=result_state.get("decision", ""),
                expected=expected_decision,
            )

        run_state = update_state(run_state, app_id, passed, lesson)
        save_run_state(run_state, state_path)

        remaining = [x for x in pending if x != app_id]
        pending_path.write_text(json.dumps(remaining))

        if run_state["decisions_since_last_improvement"] >= HILL_CLIMB_EVERY:
            system_prompt = run_hill_climbing(system_prompt, lessons, meter)
            run_state["decisions_since_last_improvement"] = 0
            save_run_state(run_state, state_path)

        if once and not remaining:
            break
