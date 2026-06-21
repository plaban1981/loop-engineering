# underwriting_loop

Self-improving insurance underwriting agent — LangGraph + Claude SDK.

## Running

```bash
cd underwriting_loop
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...

# Single application (debug)
python -m src.main --mode single --app-id APP-001

# Full event loop (processes pending_applications.json, hill-climbs every 8)
python -m src.main --mode event-loop --once

# Standalone hill-climbing (5 rounds on train split)
python -m src.main --mode improve --rounds 5
```

## Tests

```bash
pytest tests/ -v
```

Unit tests are fully offline (mocked model). Integration tests require ANTHROPIC_API_KEY.

## Budget

Hard ceiling: $2.00 per run. Raises `BudgetExhaustedError` if exceeded.
