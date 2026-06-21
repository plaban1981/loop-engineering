import json
from pathlib import Path
from langchain_core.tools import tool

_DATA = Path(__file__).parent.parent / "data" / "applications.json"


@tool
def lookup_application(app_id: str) -> dict:
    """Return the full application record for the given application ID (e.g. 'APP-001')."""
    records = json.loads(_DATA.read_text())
    for rec in records:
        if rec["id"] == app_id:
            return rec
    raise ValueError(f"Unknown application ID: {app_id}")

