from typing import Literal
from langchain_core.tools import tool


@tool
def draft_decision(
    decision: Literal["accept", "modify", "decline", "refer"],
    rationale: str,
    flags: list[str],
    recommended_premium_adj: float,
) -> dict:
    """Finalize the underwriting decision with structured output.
    decision: accept|modify|decline|refer
    rationale: written explanation referencing specific risk factors
    flags: list of applicable flags e.g. flood_rider_required, excess_claims, poor_credit, high_coverage, windstorm_exclusion
    recommended_premium_adj: fractional surcharge e.g. 0.15 for +15%
    """
    return {
        "decision": decision,
        "rationale": rationale,
        "flags": flags,
        "recommended_premium_adj": recommended_premium_adj,
    }
