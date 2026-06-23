import json
import re
from typing import Any

UNDERWRITING_RUBRIC = {
    "decision_valid": {
        "check": lambda out, app: out.get("decision") in {"accept", "modify", "decline", "refer"},
        "message": "decision must be one of: accept | modify | decline | refer",
    },
    "rationale_present": {
        "check": lambda out, app: bool(out.get("rationale", "").strip()),
        "message": "rationale must reference at least one risk factor",
    },
    "flood_rider_required": {
        "check": lambda out, app: not (
            app.get("flood_zone") and not app.get("has_flood_rider")
            and "flood_rider_required" not in out.get("flags", [])
        ),
        "message": "flood_rider_required flag missing for flood zone property without rider",
    },
    "flood_decision": {
        "check": lambda out, app: not (
            app.get("flood_zone") and not app.get("has_flood_rider")
            and out.get("decision") not in {"modify", "refer", "decline"}
        ),
        "message": "flood zone without rider requires decision of modify, refer, or decline",
    },
    "excess_claims_flag": {
        "check": lambda out, app: not (
            app.get("claim_history_5yr", 0) > 2
            and "excess_claims" not in out.get("flags", [])
        ),
        "message": "excess_claims flag required when claim_history_5yr > 2",
    },
    "excess_claims_decision": {
        "check": lambda out, app: not (
            app.get("claim_history_5yr", 0) > 2
            and out.get("decision") not in {"decline", "refer"}
        ),
        "message": "decision must be decline or refer when claim_history_5yr > 2",
    },
    "poor_credit": {
        "check": lambda out, app: not (
            app.get("credit_score_tier") == "poor"
            and out.get("decision") == "accept"
            and out.get("recommended_premium_adj", 0.0) < 0.15
        ),
        "message": "poor credit tier requires recommended_premium_adj >= 0.15 or decision decline/refer",
    },
    "high_coverage": {
        "check": lambda out, app: not (
            app.get("coverage_amount", 0) > 1_000_000
            and out.get("decision") != "refer"
        ),
        "message": "coverage > $1M requires decision: refer",
    },
    "windstorm_exclusion": {
        "check": lambda out, app: not (
            app.get("state") in {"FL", "TX"}
            and "windstorm_exclusion" not in out.get("flags", [])
        ),
        "message": "windstorm_exclusion flag required for FL/TX properties",
    },
}


def check_rubric(output: dict, rubric: dict, app: dict = None) -> list[str]:
    """Return list of violation messages. Empty list = rubric passed."""
    if app is None:
        app = output.get("app", output.get("_app_context", {}))
    return [
        rule["message"]
        for rule in rubric.values()
        if not rule["check"](output, app)
    ]


def _parse_output(raw_output: Any) -> dict:
    """Extract decision fields from agent output."""
    if isinstance(raw_output, dict):
        # Check if it already has decision field directly
        if "decision" in raw_output and raw_output.get("decision") in {"accept", "modify", "decline", "refer"}:
            return raw_output
        text = raw_output.get("output", str(raw_output))
    else:
        text = str(raw_output)

    # Try JSON extraction
    match = re.search(r'\{[^{}]*"decision"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: keyword scan
    decision = "ran_out_of_steps"
    for word in ["decline", "refer", "modify", "accept"]:
        if word in text.lower():
            decision = word
            break

    flags = re.findall(r'(flood_rider_required|excess_claims|poor_credit|high_coverage|windstorm_exclusion)', text)
    return {
        "decision": decision,
        "rationale": text[:200],
        "flags": list(set(flags)),
        "recommended_premium_adj": 0.15 if "surcharge" in text.lower() else 0.0,
    }


class RubricMiddleware:
    """Wraps a LangChain Runnable. On each invoke, checks rubric and retries up to max_retries."""

    def __init__(self, agent, rubric: dict, max_retries: int = 3):
        self._agent = agent
        self._rubric = rubric
        self._max_retries = max_retries

    def invoke(self, inputs: dict) -> dict:
        app_context = inputs.get("_app_context", {})
        current_inputs = dict(inputs)

        for attempt in range(self._max_retries + 1):
            raw = self._agent.invoke(current_inputs)
            parsed = _parse_output(raw)
            violations = check_rubric(parsed, self._rubric, app_context)

            if not violations or attempt >= self._max_retries:
                parsed["_rubric_violations"] = violations
                parsed["_attempts"] = attempt + 1
                return parsed

            # Inject feedback for retry
            violation_text = "\n".join(f"- {v}" for v in violations)
            feedback = (
                f"\nYour previous response violated the underwriting rubric:\n{violation_text}\n"
                f"Please review the application and call draft_decision again with corrections."
            )
            current_inputs = {**inputs, "input": inputs["input"] + feedback}

        parsed["_rubric_violations"] = violations
        parsed["_attempts"] = self._max_retries + 1
        return parsed
