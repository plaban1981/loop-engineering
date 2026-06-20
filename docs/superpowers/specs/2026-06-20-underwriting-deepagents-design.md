# Underwriting DeepAgents — Design Spec
**Date:** 2026-06-20  
**Project:** `underwriting_deepagents`  
**Stack:** LangChain deepagents (`create_agent`, `RubricMiddleware`) + Anthropic SDK (claude-sonnet-4-6)  
**Source concepts:** The Art of Loop Engineering (LangChain) — loops implemented at the framework abstraction level  
**Companion project:** `underwriting_loop` (LangGraph, full manual control)

---

## 1. Goal

Implement the same four-level insurance underwriting loop as `underwriting_loop`, but using LangChain deepagents' higher-level primitives. `create_agent` replaces the manual LangGraph `StateGraph`, and `RubricMiddleware` replaces the manual verification graph. The result is ~200 fewer lines of boilerplate at the cost of less visibility into loop internals.

This project serves as a direct comparison to `underwriting_loop`: same domain, same data, same tools, same judge — different abstraction level.

**Same success criteria as `underwriting_loop`:**
- Agent starts at ~30–50% accuracy on train split
- Hill-climbing loop reaches ≥80% on train split within 5 rounds
- Improved prompt generalizes to ≥75% on held-out test split
- Full run costs under $2.00
- Evolved prompt passes the anti-cheat auditor

---

## 2. Project Layout

```
underwriting_deepagents/
├── src/
│   ├── data/                         # Identical to underwriting_loop — shared JSON files
│   │   ├── applications.json
│   │   ├── rules_kb.json
│   │   └── historical_decisions.json
│   ├── tools/                        # Same 4 tools as underwriting_loop
│   │   ├── application_tool.py
│   │   ├── rules_tool.py
│   │   ├── risk_signal_tool.py
│   │   └── decision_tool.py
│   ├── agent.py                      # create_agent setup + rubric definition
│   ├── middleware.py                 # RubricMiddleware wired to deterministic judge
│   ├── event_loop.py                 # Queue poller — fires verified_agent per application
│   ├── hill_climbing.py              # Reflect → mutate → evaluate → gate
│   ├── judge.py                      # Deterministic decision checker (no LLM)
│   ├── memory.py                     # Online lesson distillation
│   ├── billing.py                    # Metered Anthropic client + budget kill switch
│   ├── audit.py                      # Judge checksum + prompt auditor
│   └── main.py                       # CLI entry point (3 run modes)
├── state/
│   ├── pending_applications.json     # Event queue
│   ├── run_state.json                # Processed/pending/failed tracker + lessons
│   └── best_prompt.txt               # Current best system prompt
├── traces/                           # Per-application run traces (JSON)
├── pyproject.toml
└── CLAUDE.md
```

**What is shared with `underwriting_loop` (identical files, not copied — symlinked or duplicated):**
- `src/data/` — all three JSON data files
- `src/tools/` — all four tool functions
- The judge logic, memory distillation pattern, audit defenses, and cost metering are re-implemented in this project (same logic, different module paths) so each project is self-contained and runnable independently.

---

## 3. Data Model

Identical to `underwriting_loop`. See Section 4 of the LangGraph design spec for full field definitions.

**applications.json:** 20 records, 16 train / 4 test (APP-001–016 train, APP-017–020 test).  
**rules_kb.json:** 5 hidden underwriting conventions the seed prompt does not mention.  
**historical_decisions.json:** Ground-truth decisions keyed by application ID with train/test split.

---

## 4. Tools

Identical to `underwriting_loop`. Four tool functions:

| Tool | Signature | Returns |
|------|-----------|---------|
| `lookup_application` | `(app_id: str)` | Full application dict |
| `retrieve_rules` | `(risk_factor: str)` | List of matching rule strings |
| `fetch_risk_signal` | `(app_id: str, signal_type: str)` | Simulated API dict |
| `draft_decision` | `(risk_score: float, flags: list[str])` | `DecisionOutput` (Pydantic model) |

Tools are imported directly by `agent.py` and passed to `create_agent`.

---

## 5. Loop Implementations

### 5.1 Loop 1 + Loop 2 — Agent + Verification (`agent.py`, `middleware.py`)

In `underwriting_loop` these are two separate LangGraph graphs (~120 lines). Here they collapse into `create_agent` + `RubricMiddleware`:

**`agent.py`:**
```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

def build_agent(system_prompt: str, metered_client) -> Runnable:
    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_client=metered_client,
    )
    return create_agent(
        model=model,
        tools=[lookup_application, retrieve_rules, fetch_risk_signal, draft_decision],
        system_prompt=system_prompt,
    )
```

`create_agent` handles the ReAct loop internally: model → tool_calls → tool results → model → … → final answer. `max_iterations` is set to 8 (same step budget as Project A).

**`middleware.py`:**
```python
from langchain.agents.middleware import RubricMiddleware

UNDERWRITING_RUBRIC = """
The underwriting decision must satisfy ALL of the following:
1. decision field is one of: accept | modify | decline | refer
2. rationale references at least one specific risk factor from the application
3. If application.flood_zone is true and application.has_flood_rider is false,
   flags must include "flood_rider_required" and decision must be "modify" or "refer"
4. If application.claim_history_5yr > 2, flags must include "excess_claims"
   and decision must be "decline" or "refer"
5. If application.credit_score_tier is "poor", recommended_premium_adj >= 0.15
   or decision must be "decline"
6. If application.coverage_amount > 1000000, decision must be "refer"
7. If application.state is "FL" or "TX", flags must include "windstorm_exclusion"
"""

def build_verified_agent(system_prompt: str, metered_client) -> Runnable:
    agent = build_agent(system_prompt, metered_client)
    return RubricMiddleware(
        agent,
        rubric=UNDERWRITING_RUBRIC,
        max_retries=3,
    )
```

`RubricMiddleware` wraps the agent. On each run it:
1. Invokes the agent
2. Checks output against the rubric (using the framework's internal grader)
3. If rubric fails and `retry_count < max_retries`: re-invokes agent with rubric feedback appended
4. If rubric passes or `max_retries` exhausted: returns final output

**What `RubricMiddleware` does NOT do (vs. Project A's manual grader):**
- Does not check against `historical_decisions.json` ground truth — rubric is structural/rule-based, not outcome-based
- Does not produce a numeric score — pass/fail per rubric item only
- Does not route `"refer"` decisions to a human queue — that is handled by `event_loop.py` after the fact

**Human-in-the-loop:** `event_loop.py` inspects the final decision after `verified_agent` completes. If `decision == "refer"`, it writes to `state/human_queue.json` and skips judge scoring for that application.

### 5.2 Loop 3 — Event Loop (`event_loop.py`)

Identical in logic to `underwriting_loop`. Polls `state/pending_applications.json`, fires `verified_agent` (Loop 1+2) for each application, persists results to `state/run_state.json`.

```python
def run_event_loop(verified_agent, judge, memory, once=False):
    while True:
        pending = load_pending()
        if not pending:
            if once:
                break
            time.sleep(POLL_INTERVAL)
            continue

        app_id = pending[0]
        result = verified_agent.invoke({"input": f"Evaluate application {app_id}"})
        decision = extract_decision(result)

        passed = judge.is_correct(decision, app_id)
        if not passed:
            lesson = memory.distill(app_id, decision)
            memory.append(lesson)

        log_trace(app_id, result, passed)
        update_state(app_id, passed)

        if decisions_since_improvement() >= 8:
            run_hill_climbing(judge, memory)
```

State file schema is identical to `underwriting_loop/state/run_state.json`.

### 5.3 Loop 4 — Hill Climbing (`hill_climbing.py`)

Same reflect → mutate → evaluate → gate pattern as `underwriting_loop`. The only difference: `evaluate()` internally calls `build_verified_agent(candidate_prompt, metered_client)` instead of a LangGraph graph runner.

```python
def improve(seed_prompt: str, rounds: int = 5) -> str:
    best = seed_prompt
    best_score, failures = judge.evaluate(build_verified_agent, best, split="train")

    for r in range(1, rounds + 1):
        if not failures:
            break
        candidate = reflect(best, failures)                          # mutate
        score, cand_failures = judge.evaluate(build_verified_agent, candidate)  # evaluate
        verdict = audit.audit_prompt(candidate)                       # anti-cheat
        if score > best_score and verdict == "VERDICT: CLEAN":       # gate
            best, best_score, failures = candidate, score, cand_failures
            write_best_prompt(best)
            print(f"round {r}: {score:.0%} KEPT")
        else:
            print(f"round {r}: {score:.0%} reverted")

    return best
```

**Reflect prompt:** Identical to `underwriting_loop` — same `REFLECT` template, same "do not memorize specific app IDs" instruction.

---

## 6. Judge (`judge.py`)

Identical logic to `underwriting_loop`. Never calls an LLM.

```python
def is_correct(decision: str, app_id: str) -> bool:
    expected = load_historical_decisions()[app_id]["expected_decision"]
    return decision == expected

def evaluate(agent_factory, system_prompt: str, split: str = "train") -> tuple[float, list]:
    tasks = load_tasks(split)
    failures = []
    for task in tasks:
        agent = agent_factory(system_prompt, metered_client)
        result = agent.invoke({"input": f"Evaluate application {task['id']}"})
        decision = extract_decision(result)
        if not is_correct(decision, task["id"]):
            failures.append({
                "app_id": task["id"],
                "got": decision,
                "expected": task["expected_decision"],
                "application": task["application"],
            })
    score = (len(tasks) - len(failures)) / len(tasks)
    return score, failures
```

**Key difference from Project A:** `evaluate()` accepts `agent_factory` as a parameter (callable that returns `verified_agent`) instead of a raw system prompt runner, because `create_agent` must be re-instantiated per evaluation run.

---

## 7. Memory (`memory.py`)

Identical distillation logic to `underwriting_loop`. Same `DISTILL_PROMPT`, same lesson list, same gain measurement protocol (stateful vs. stateless on held-out test split).

```python
def with_lessons(base_prompt: str, lessons: list[str]) -> str:
    if not lessons:
        return base_prompt
    return base_prompt + "\n\nUnderwriting lessons learned:\n" + \
           "\n".join(f"- {l}" for l in lessons)
```

---

## 8. Defenses (`audit.py`)

Identical to `underwriting_loop`:

| Defense | Implementation |
|---------|---------------|
| Train/test split | Same 16/4 split; Loop 4 only sees train failures |
| Judge checksum | SHA-256 of `historical_decisions.json` verified before each score counts |
| Prompt auditor | Separate `claude-sonnet-4-6` call reads evolved prompt; `VERDICT: CLEAN` or `VERDICT: CHEATING` |

---

## 9. Cost Metering (`billing.py`)

Identical to `underwriting_loop`. Wraps the Anthropic client, hard ceiling at `$2.00`, prints receipt at end of run.

---

## 10. Entry Point (`main.py`)

Same three run modes as `underwriting_loop`:

```bash
# Single application (debug mode)
python -m src.main --mode single --app-id APP-001

# Full event loop
python -m src.main --mode event-loop [--once]

# Standalone hill-climbing
python -m src.main --mode improve --rounds 5
```

---

## 11. Key Differences vs. `underwriting_loop` (Summary)

| Aspect | `underwriting_loop` (LangGraph) | `underwriting_deepagents` (deepagents) |
|--------|--------------------------------|----------------------------------------|
| Loop 1 implementation | Manual `StateGraph` + `agent_node` + `tools_node` + conditional edges | `create_agent(model, tools)` |
| Loop 2 implementation | Manual `grader_node` + retry graph + `UnderwritingState.retry_count` | `RubricMiddleware(agent, rubric, max_retries=3)` |
| State management | `UnderwritingState` TypedDict passed through graph | Managed internally by framework |
| Tool call visibility | Full — each tool call is a named graph node transition | Partial — framework logs tool calls but they are not directly inspectable in Python |
| Verification feedback format | Structured `HumanMessage` with specific field hints | Framework-managed retry prompt (rubric items that failed) |
| Loop 3 | Custom queue poller | Same custom queue poller |
| Loop 4 | Same reflect/mutate/gate pattern | Same reflect/mutate/gate pattern |
| Boilerplate | ~350 lines for loops 1+2 | ~150 lines for loops 1+2 |
| Customisability | Full | Limited to what `create_agent`/`RubricMiddleware` expose |

---

## 12. Dependencies

```toml
[project]
name = "underwriting_deepagents"
requires-python = ">=3.10"

dependencies = [
    "anthropic>=0.40.0",
    "langchain>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-deepagents>=0.1.0",
    "pydantic>=2.0",
]
```

---

## 13. Out of Scope

Same as `underwriting_loop`: no real insurance data, no fine-tuning, no UI, no LangSmith deployment, no production infrastructure.
