import os
import anthropic
from .judge import evaluate

DISTILL_PROMPT = """An insurance underwriting agent made a wrong decision.

Application details: {application}
Agent decided: {got}
Correct decision: {expected}

Write ONE short, general underwriting rule that would prevent this mistake.
Do NOT reference this specific application ID or applicant details.
Reply with only the rule, as a single sentence."""


def with_lessons(base_prompt: str, lessons: list[str]) -> str:
    if not lessons:
        return base_prompt
    return base_prompt + "\n\nUnderwriting lessons:\n" + "\n".join(f"- {l}" for l in lessons)


def distill_lesson(application: dict, got: str, expected: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": DISTILL_PROMPT.format(
            application=application, got=got, expected=expected
        )}],
    )
    return resp.content[0].text.strip()


def measure_gain(base_prompt: str, lessons: list[str], agent_factory, split: str = "test", meter=None) -> float:
    stateless_score, _ = evaluate(agent_factory, base_prompt, split=split, meter=meter)
    stateful_score, _ = evaluate(agent_factory, with_lessons(base_prompt, lessons), split=split, meter=meter)
    return stateful_score - stateless_score
