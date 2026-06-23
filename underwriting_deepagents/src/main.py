import argparse
import json
import os
from pathlib import Path

from .agent import SEED_PROMPT, build_verified_agent
from .audit import compute_judge_checksum
from .billing import CostMeter
from .event_loop import run_event_loop
from .hill_climbing import improve
from .judge import evaluate

_STATE_DIR = Path(__file__).parent.parent / "state"


def _init_run_state() -> dict:
    path = _STATE_DIR / "run_state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state["judge_checksum"] = compute_judge_checksum()
    path.write_text(json.dumps(state, indent=2))
    return state


def cmd_single(args):
    meter = CostMeter()
    best_path = _STATE_DIR / "best_prompt.txt"
    prompt = best_path.read_text().strip() if best_path.exists() else SEED_PROMPT

    from .tools.application_tool import lookup_application
    app_context = lookup_application.invoke({"app_id": args.app_id})
    agent = build_verified_agent(prompt, meter)
    result = agent.invoke({
        "input": f"Evaluate insurance application {args.app_id}. Use all tools and call draft_decision.",
        "_app_context": app_context,
    })
    print(f"Decision:  {result.get('decision')}")
    print(f"Flags:     {result.get('flags', [])}")
    print(f"Attempts:  {result.get('_attempts', 1)}")
    meter.print_receipt()


def cmd_event_loop(args):
    meter = CostMeter()
    _init_run_state()
    best_path = _STATE_DIR / "best_prompt.txt"
    prompt = best_path.read_text().strip() if best_path.exists() else SEED_PROMPT
    run_event_loop(system_prompt=prompt, meter=meter, once=args.once)
    meter.print_receipt()


def cmd_improve(args):
    meter = CostMeter()
    _init_run_state()
    run_state = json.loads((_STATE_DIR / "run_state.json").read_text())
    lessons = run_state.get("lessons", [])
    best_path = _STATE_DIR / "best_prompt.txt"
    seed = best_path.read_text().strip() if best_path.exists() else SEED_PROMPT

    best = improve(build_verified_agent, seed, lessons, meter, rounds=args.rounds)
    best_path.write_text(best)

    print("\nFinal test evaluation...")
    seed_score, _ = evaluate(build_verified_agent, SEED_PROMPT, split="test", meter=meter)
    final_score, _ = evaluate(build_verified_agent, best, split="test", meter=meter)
    meter.print_receipt(test_score=final_score, seed_test_score=seed_score)


def main():
    parser = argparse.ArgumentParser(description="Underwriting DeepAgents")
    parser.add_argument("--mode", choices=["single", "event-loop", "improve"], required=True)
    parser.add_argument("--app-id")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "single":
        if not args.app_id:
            parser.error("--app-id required")
        cmd_single(args)
    elif args.mode == "event-loop":
        cmd_event_loop(args)
    elif args.mode == "improve":
        cmd_improve(args)


if __name__ == "__main__":
    main()
