from langchain_core.tools import tool
from .application_tool import lookup_application


@tool
def fetch_risk_signal(app_id: str, signal_type: str) -> dict:
    """Fetch simulated external risk signal. signal_type: 'credit' | 'flood' | 'property' | 'claims'"""
    app = lookup_application.invoke({"app_id": app_id})
    if signal_type == "credit":
        tier = app["credit_score_tier"]
        return {
            "score": {"good": 720, "fair": 650, "poor": 575}[tier],
            "tier": tier,
            "risk_label": {"good": "low", "fair": "medium", "poor": "high"}[tier],
        }
    if signal_type == "flood":
        return {
            "flood_zone": app["flood_zone"],
            "zone_code": "AE" if app["flood_zone"] else "X",
            "flood_risk_score": 0.82 if app["flood_zone"] else 0.08,
        }
    if signal_type == "property":
        cov = app["coverage_amount"]
        return {
            "estimated_value": int(cov * 0.95),
            "replacement_cost": int(cov * 1.05),
            "construction_type": "frame" if cov < 500_000 else "masonry",
        }
    if signal_type == "claims":
        count = app["claim_history_5yr"]
        return {
            "claim_count_5yr": count,
            "largest_claim_amt": count * 15_000 if count > 0 else 0,
            "claim_types": (["water_damage"] * min(count, 2) + ["wind"] * max(0, count - 2)),
        }
    raise ValueError(f"Unknown signal type: {signal_type!r}. Use: credit|flood|property|claims")
