import json
from pathlib import Path
from langchain_core.tools import tool

_DATA = Path(__file__).parent.parent / "data" / "rules_kb.json"


@tool
def retrieve_rules(risk_factor: str) -> list[str]:
    """Return underwriting rules matching the given risk factor keyword."""
    kb = json.loads(_DATA.read_text())
    keyword = risk_factor.lower()
    matches = [
        rule["text"]
        for rule in kb["rules"]
        if any(kw.lower() in keyword or keyword in kw.lower() for kw in rule["keywords"])
    ]
    return matches if matches else ["No specific rules found for this risk factor."]
