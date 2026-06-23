import os
from pathlib import Path
from typing import Callable

import anthropic

from .audit import audit_prompt
from .judge import evaluate
from .memory import with_lessons

_STATE_DIR = Path(__file__).parent.parent / "state"
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
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    failure_text = "\n".join(
        f"- {f['app_id']}: said '{f['got']}', expected '{f['expected']}'. App: {f['application']}"
        for f in failures
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": REFLECT_TEMPLATE.format(
            prompt=current_prompt,
            failures=failure_text,
        )}],
    )
    return resp.content[0].text.strip()


def improve(
    agent_factory: Callable,
    seed_prompt: str,
    lessons: list[str],
    meter,
    rounds: int = 5,
) -> str:
    best = with_lessons(seed_prompt, lessons)
    best_score, failures = evaluate(agent_factory, best, split="train", meter=meter)
    print(f"seed score: {best_score:.0%}  ({len(failures)} failures)")

    for r in range(1, rounds + 1):
        if not failures:
            print(f"round {r}: 100% — stopping")
            break

        candidate = reflect(best, failures)
        score, cand_failures = evaluate(agent_factory, candidate, split="train", meter=meter)
        verdict = audit_prompt(candidate)

        if score > best_score and verdict == "VERDICT: CLEAN":
            best, best_score, failures = candidate, score, cand_failures
            BEST_PROMPT_PATH.write_text(best, encoding="utf-8")
            print(f"round {r}: {score:.0%} KEPT  ({len(cand_failures)} failures)")
        else:
            reason = "cheating" if verdict != "VERDICT: CLEAN" else f"{score:.0%} <= {best_score:.0%}"
            print(f"round {r}: {score:.0%} REVERTED ({reason})")

    return best
