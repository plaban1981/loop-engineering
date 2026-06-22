# src/memory/lesson_memory.py
import os
import anthropic

from ..judge.judge import evaluate_prompt

DISTILL_PROMPT = """An insurance underwriting agent made a wrong decision.

Application details: {application}
Agent decided: {got}
Correct decision: {expected}

Write ONE short, general underwriting rule that would prevent this mistake.
Do NOT reference this specific application ID or any applicant names/details.
Reply with only the rule, as a single sentence."""


def with_lessons(base_prompt: str, lessons: list[str]) -> str:
    if not lessons:
        return base_prompt
    lessons_text = "\n".join(f"- {l}" for l in lessons)
    return base_prompt + f"\n\nUnderwriting lessons learned:\n{lessons_text}"


def distill_lesson(application: dict, got: str, expected: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": DISTILL_PROMPT.format(
                application=application,
                got=got,
                expected=expected,
            ),
        }],
    )
    return response.content[0].text.strip()


def measure_gain(
    base_prompt: str,
    lessons: list[str],
    split: str = "test",
    meter=None,
) -> float:
    """CL-BENCH protocol: stateful_score - stateless_score on held-out split."""
    stateless_score, _ = evaluate_prompt(base_prompt, split=split, meter=meter)
    augmented = with_lessons(base_prompt, lessons)
    stateful_score, _ = evaluate_prompt(augmented, split=split, meter=meter)
    return stateful_score - stateless_score
