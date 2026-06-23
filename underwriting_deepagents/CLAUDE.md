# underwriting_deepagents

Self-improving insurance underwriting agent — LangChain create_react_agent + RubricMiddleware.
Companion project to underwriting_loop (LangGraph). Same domain, same data, higher abstraction.

## Running

```bash
cd underwriting_deepagents
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...

python -m src.main --mode single --app-id APP-001
python -m src.main --mode event-loop --once
python -m src.main --mode improve --rounds 5
```

## Design notes

- `RubricMiddleware` is implemented as a custom class (not a published package).
  It wraps any LangChain Runnable and retries up to max_retries times if the
  structural rubric check fails.
- `create_react_agent` is from `langgraph.prebuilt` (not `langchain.agents`).
- Judge's evaluate() receives agent_factory (callable) because create_react_agent
  objects must be re-instantiated per eval run.
