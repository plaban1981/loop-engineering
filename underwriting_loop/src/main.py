# src/main.py
import argparse
import json
import os
from pathlib import Path

from .billing.cost_meter import CostMeter
from .judge.audit import compute_judge_checksum
from .judge.judge import evaluate_prompt
from .loops.loop1_agent import SEED_PROMPT, run_loop1
from .loops.loop3_event import run_event_loop
from .loops.loop4_hill_climbing import improve
from .memory.lesson_memory import with_lessons

_STATE_DIR = Path(__file__).parent.parent / "state"


_DEFAULT_RUN_STATE = {
    "processed": [], "passed": [], "failed": [], "pending": [],
    "lessons": [], "decisions_since_last_improvement": 0, "judge_checksum": ""
}


def _init_run_state() -> dict:
    path = _STATE_DIR / "run_state.json"
    state = {}
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                state = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    if not state:
        state = dict(_DEFAULT_RUN_STATE)
    state["judge_checksum"] = compute_judge_checksum()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def cmd_single(args):
    meter = CostMeter()
    best_path = _STATE_DIR / "best_prompt.txt"
    prompt = best_path.read_text().strip() if best_path.exists() else SEED_PROMPT

    print(f"Running single application: {args.app_id}")
    state = run_loop1(args.app_id, prompt, meter)
    print(f"Decision:  {state['decision']}")
    print(f"Flags:     {state.get('flags', [])}")
    print(f"Rationale: {state.get('rationale', '')[:300]}")
    meter.print_receipt()


def cmd_event_loop(args):
    meter = CostMeter()
    _init_run_state()
    best_path = _STATE_DIR / "best_prompt.txt"
    prompt = best_path.read_text().strip() if best_path.exists() else SEED_PROMPT

    run_event_loop(
        system_prompt=prompt,
        meter=meter,
        once=args.once,
    )
    meter.print_receipt()


def cmd_improve(args):
    meter = CostMeter()
    run_state = _init_run_state()
    lessons = run_state.get("lessons", [])

    best_path = _STATE_DIR / "best_prompt.txt"
    seed = best_path.read_text().strip() if best_path.exists() else SEED_PROMPT

    print(f"Starting hill-climbing ({args.rounds} rounds)...")
    best = improve(seed, lessons, meter, rounds=args.rounds)
    best_path.write_text(best)

    print("\nEvaluating on test split...")
    seed_score, _ = evaluate_prompt(SEED_PROMPT, split="test", meter=meter)
    final_score, _ = evaluate_prompt(best, split="test", meter=meter)
    meter.print_receipt(
        train_score=None,
        test_score=final_score,
        seed_test_score=seed_score,
    )


def main():
    parser = argparse.ArgumentParser(description="Underwriting Loop Agent")
    parser.add_argument("--mode", choices=["single", "event-loop", "improve"], required=True)
    parser.add_argument("--app-id", help="Application ID for --mode single")
    parser.add_argument("--once", action="store_true", help="Process queue once then exit (event-loop mode)")
    parser.add_argument("--rounds", type=int, default=5, help="Hill-climbing rounds (improve mode)")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.app_id:
            parser.error("--app-id required for --mode single")
        cmd_single(args)
    elif args.mode == "event-loop":
        cmd_event_loop(args)
    elif args.mode == "improve":
        cmd_improve(args)


if __name__ == "__main__":
    main()
