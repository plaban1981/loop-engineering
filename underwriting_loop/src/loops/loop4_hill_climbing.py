# src/loops/loop4_hill_climbing.py
import os
from pathlib import Path

import anthropic

from ..billing.cost_meter import CostMeter
from ..judge.audit import audit_prompt
from ..judge.judge import evaluate_prompt
from ..memory.lesson_memory import with_lessons

_STATE_DIR = Path(__file__).parent.parent.parent / "state"
BEST_PROMPT_PATH = _STATE_DIR / "best_prompt.txt"

REFLECT_TEMPLATE = """You are improving the system prompt of an insurance underwriting agent.

CURRENT PROMPT:
{prompt}

The agent got these decisions WRONG on training applications:
{failures}

Look for PATTERNS in the failures. Infer GENERAL underwriting rules that would fix them.
Do NOT memorize specific application IDs or applicant details.
Write an improved system prompt. Reply with ONLY the new prompt text."""


def reflect(current_prompt: str, failures: list[dict]) -> str:
    client = anthropic.Anthropic()
    failure_text = "\n".join(
        f"- APP {f['app_id']}: agent said '{f['got']}', expected '{f['expected']}'. "
        f"Application data: {f['application']}"
        for f in failures
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": REFLECT_TEMPLATE.format(
                prompt=current_prompt,
                failures=failure_text,
            ),
        }],
    )
    return response.content[0].text.strip()


def improve(
    seed_prompt: str,
    lessons: list[str],
    meter: CostMeter,
    rounds: int = 5,
) -> str:
    """Hill-climbing: reflect -> mutate -> evaluate -> gate. Returns best prompt found."""
    best = with_lessons(seed_prompt, lessons)
    best_score, failures = evaluate_prompt(best, split="train", meter=meter)
    print(f"seed score: {best_score:.0%}  ({len(failures)} failures)")

    for r in range(1, rounds + 1):
        if not failures:
            print(f"round {r}: 100% — stopping early")
            break

        candidate = reflect(best, failures)
        score, cand_failures = evaluate_prompt(candidate, split="train", meter=meter)
        verdict = audit_prompt(candidate)

        if score > best_score and verdict == "VERDICT: CLEAN":
            best, best_score, failures = candidate, score, cand_failures
            BEST_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            BEST_PROMPT_PATH.write_text(best, encoding="utf-8")
            print(f"round {r}: {score:.0%} KEPT  ({len(cand_failures)} failures)")
        else:
            reason = "cheating" if verdict != "VERDICT: CLEAN" else f"score {score:.0%} <= best {best_score:.0%}"
            print(f"round {r}: {score:.0%} REVERTED ({reason})")

    return best
