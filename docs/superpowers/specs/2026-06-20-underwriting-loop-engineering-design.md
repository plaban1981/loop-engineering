# Underwriting Loop Engineering — Design Spec
**Date:** 2026-06-20  
**Project:** `underwriting_loop`  
**Stack:** LangGraph + Anthropic SDK (claude-sonnet-4-6)  
**Source concepts:** The Art of Loop Engineering (LangChain), AddyOsmani Loop Engineering, RSI Book Ch1 & Ch2

---

## 1. Goal

Build a self-improving insurance underwriting agent that implements all four loop levels from the LangChain "Art of Loop Engineering" framework, applied to a full intake → score → decision pipeline. The agent starts with a generic seed prompt, makes systematic errors on hidden underwriting conventions, and progressively corrects itself through reflective prompt mutation gated by a deterministic judge.

Success criteria:
- Agent starts at ~30–50% accuracy on train split (seed prompt, no domain knowledge)
- Hill-climbing loop reaches ≥80% on train split within 5 rounds
- Improved prompt generalizes: ≥75% on held-out test split (rule learning, not memorization)
- Full run costs under $2.00 (budget kill switch enforces this)
- Evolved prompt passes the anti-cheat auditor (no hardcoded app IDs or decisions)

---

## 2. Project Layout

```
underwriting_loop/
├── src/
│   ├── data/
│   │   ├── applications.json           # 20 structured application records (16 train / 4 test)
│   │   ├── rules_kb.json               # Underwriting rules knowledge base
│   │   └── historical_decisions.json   # Ground-truth decisions with train/test split
│   ├── tools/
│   │   ├── application_tool.py         # lookup_application(app_id) → dict
│   │   ├── rules_tool.py               # retrieve_rules(risk_factor) → list[str]
│   │   ├── risk_signal_tool.py         # fetch_risk_signal(app_id, signal_type) → dict
│   │   └── decision_tool.py            # draft_decision(risk_score, flags) → DecisionOutput
│   ├── loops/
│   │   ├── loop1_agent.py              # ReAct agent (LangGraph StateGraph)
│   │   ├── loop2_verification.py       # Verification + retry graph
│   │   ├── loop3_event.py              # Event-driven queue loop
│   │   └── loop4_hill_climbing.py      # Trace analysis + prompt mutation + gate
│   ├── judge/
│   │   ├── judge.py                    # Deterministic decision checker (no LLM)
│   │   └── audit.py                    # Prompt auditor + judge checksum
│   ├── memory/
│   │   └── lesson_memory.py            # Online lesson distillation
│   ├── billing/
│   │   └── cost_meter.py               # Token cost tracker + budget kill switch
│   └── main.py                         # CLI entry point (3 run modes)
├── traces/                             # Per-application run traces (JSON)
├── state/
│   ├── pending_applications.json       # Event queue
│   ├── run_state.json                  # Processed/pending/failed tracker
│   └── best_prompt.txt                 # Current best system prompt (hill-climbing output)
├── pyproject.toml
└── CLAUDE.md
```

---

## 3. Shared State Schema

```python
class UnderwritingState(TypedDict):
    application_id: str
    application: dict           # raw application fields
    messages: list[AnyMessage]  # full agent transcript (LangGraph messages)
    risk_factors: list[str]     # identified risk factors from tool calls
    risk_score: float           # 0–100 composite risk score
    decision: str               # "accept" | "modify" | "decline" | "refer"
    rationale: str              # agent's written rationale
    flags: list[str]            # e.g. ["flood_rider_required", "excess_claims"]
    recommended_premium_adj: float  # percentage adjustment suggestion
    verification_feedback: str  # grader feedback injected on retry
    retry_count: int            # max 3
    lessons: list[str]          # distilled memory from past failures this run
    traces: list[dict]          # structured trace entries for hill-climbing
    spend: float                # running USD cost for this run
```

---

## 4. Data Design

### 4.1 applications.json

20 records. Fields per application:

| Field | Type | Example |
|-------|------|---------|
| `id` | str | `"APP-001"` |
| `split` | str | `"train"` or `"test"` |
| `applicant_age` | int | `34` |
| `state` | str | `"FL"` |
| `property_type` | str | `"residential"` |
| `coverage_amount` | int | `450000` |
| `claim_history_5yr` | int | `3` |
| `credit_score_tier` | str | `"good"` \| `"fair"` \| `"poor"` |
| `flood_zone` | bool | `true` |
| `has_flood_rider` | bool | `false` |
| `business_use` | bool | `false` |

Train split: APP-001 through APP-016 (16 records).  
Test split: APP-017 through APP-020 (4 records, unseen by improvement loop).

### 4.2 rules_kb.json — Hidden Conventions

Five underwriting conventions the seed prompt does NOT mention (the agent must learn these):

1. Flood zone property without flood rider → decision must be `"modify"` (require rider endorsement)
2. `claim_history_5yr > 2` → `"decline"` or `"refer"` to senior underwriter
3. `credit_score_tier == "poor"` → minimum 15% surcharge or `"decline"`
4. `coverage_amount > 1,000,000` → mandatory senior review (`"refer"`)
5. State `FL` or `TX` → windstorm exclusion applies by default (must be flagged)

### 4.3 historical_decisions.json — Ground Truth

```json
{
  "APP-001": {
    "split": "train",
    "expected_decision": "modify",
    "expected_flags": ["flood_rider_required", "windstorm_exclusion"]
  }
}
```

Judge checks `decision` field against `expected_decision`. Flags are checked for presence (not exact match) as a secondary score signal.

---

## 5. Tools

### 5.1 lookup_application(app_id: str) → dict
Returns the full application record from `applications.json`. Raises `ValueError` for unknown IDs.

### 5.2 retrieve_rules(risk_factor: str) → list[str]
Keyword search over `rules_kb.json`. Accepts terms like `"flood"`, `"claims"`, `"coverage"`, `"credit"`, `"state"`. Returns matching rule strings.

### 5.3 fetch_risk_signal(app_id: str, signal_type: str) → dict
Simulated external API. Deterministic: derives values from the application record so results are reproducible. Signal types:
- `"credit"` → `{score: int, tier: str, risk_label: str}`
- `"flood"` → `{flood_zone: bool, zone_code: str, flood_risk_score: float}`
- `"property"` → `{estimated_value: int, replacement_cost: int, construction_type: str}`
- `"claims"` → `{claim_count_5yr: int, largest_claim_amt: int, claim_types: list[str]}`

### 5.4 draft_decision(risk_score: float, flags: list[str]) → DecisionOutput
Structured output tool (uses Claude tool_use with a typed schema). Produces:
```python
class DecisionOutput(BaseModel):
    decision: Literal["accept", "modify", "decline", "refer"]
    rationale: str
    flags: list[str]
    recommended_premium_adj: float  # e.g. 0.15 for +15%
```

---

## 6. Loop Implementations

### 6.1 Loop 1 — Agent Loop (`loop1_agent.py`)

A LangGraph `StateGraph` implementing the ReAct pattern.

**Nodes:**
- `agent_node`: calls `claude-sonnet-4-6` with current messages + system prompt + lessons. Returns message with optional tool_calls.
- `tools_node`: dispatches tool calls to the four tool functions. Appends `ToolMessage` results to state.

**Edges:**
- `agent_node` → `tools_node` if `msg.tool_calls` is non-empty
- `agent_node` → `END` if no tool calls (agent answered)
- Hard cap: `max_steps = 8`. If exceeded, state is returned with `decision = "ran_out_of_steps"`.

**System prompt evolution:** Loop 1 reads `state/best_prompt.txt` at startup. This file is updated by Loop 4 when a better prompt is found.

### 6.2 Loop 2 — Verification Loop (`loop2_verification.py`)

Wraps Loop 1 as a subgraph.

**Nodes:**
- `run_agent`: invokes the Loop 1 subgraph, fills `decision`, `rationale`, `flags` in state.
- `grader_node`: deterministic check. Calls `judge.is_correct(state.decision, expected)`. If fail and `retry_count < 3`: appends structured feedback to messages, increments `retry_count`, returns to `run_agent`. If pass or `retry_count == 3`: proceeds to `END`.

**Grader feedback format (injected as HumanMessage on retry):**
```
"Your decision was '{got}' but this application requires '{expected}'.
Review the following risk factors and try again: {failure_hints}"
```

**Human-in-the-loop:** `"refer"` decisions skip the grader entirely and write to `state/human_queue.json`. A human can then approve/reject from the queue file.

**Verification outcome logged to trace:**
```python
{"app_id": "APP-001", "attempt": 1, "decision": "accept", "pass": False,
 "feedback": "...", "spend_usd": 0.012}
```

### 6.3 Loop 3 — Event Loop (`loop3_event.py`)

Polls `state/pending_applications.json`. Fires Loop 2 for each pending application. Persists state between runs.

**State file (`run_state.json`) schema:**
```json
{
  "processed": ["APP-001", "APP-002"],
  "passed": ["APP-002"],
  "failed": ["APP-001"],
  "pending": ["APP-003", "APP-004"],
  "lessons": ["Flood zone without rider → modify"],
  "decisions_since_last_improvement": 4
}
```

**Behavior:**
- After processing each application: update `run_state.json`, append lesson if verification failed.
- Every 8 processed applications: trigger Loop 4.
- Empty queue: print summary and exit (or sleep and re-poll in production mode).

**Run modes:**
- `--mode event-loop`: polls indefinitely (production)
- `--mode event-loop --once`: processes current queue and exits (testing)

### 6.4 Loop 4 — Hill Climbing Loop (`loop4_hill_climbing.py`)

Fires after every 8 decisions. Mutates the system prompt using reflective failure analysis, gates the change.

**Steps:**
1. **Collect traces**: read last N trace files from `traces/`.
2. **Reflect**: call `claude-sonnet-4-6` with failure report:
   ```
   "You are improving the system prompt of an insurance underwriting agent.
   CURRENT PROMPT: {prompt}
   The agent got these decisions WRONG: {failures}
   Look for patterns. Infer GENERAL underwriting rules.
   Do NOT memorize specific application IDs or applicant details.
   Write an improved system prompt. Reply with ONLY the new prompt."
   ```
3. **Evaluate**: run `judge.evaluate(candidate_prompt, split="train")` → score + failures.
4. **Gate**: if `score > best_score` → write to `state/best_prompt.txt`, update `best_score`. Else revert. Always revert to best-ever-seen, never to previous attempt.
5. **Audit**: call `audit.audit_prompt(evolved_prompt)` → `VERDICT: CLEAN` or `VERDICT: CHEATING`. Log result. If cheating detected, discard candidate regardless of score.
6. **Checksum verify**: `audit.verify_judge_checksum()` — SHA-256 of `historical_decisions.json` must match value recorded at run start.

---

## 7. Judge

**`judge.py`** — never calls an LLM.

```python
def is_correct(agent_decision: str, expected: dict) -> bool:
    return agent_decision == expected["expected_decision"]

def evaluate(system_prompt: str, split: str = "train") -> tuple[float, list[dict]]:
    tasks = load_tasks(split)
    failures = []
    for task in tasks:
        state = run_loop1(task["application_id"], system_prompt)
        if not is_correct(state["decision"], task):
            failures.append({
                "app_id": task["application_id"],
                "got": state["decision"],
                "expected": task["expected_decision"],
                "application": task["application"]
            })
    score = (len(tasks) - len(failures)) / len(tasks)
    return score, failures
```

**Train/test isolation:** `evaluate()` accepts a `split` argument. Loop 4 only ever calls `evaluate(split="train")`. Test split is only called in `main.py` to measure final generalization.

---

## 8. Memory

**`lesson_memory.py`** — online distillation, one general rule per failure.

```python
DISTILL_PROMPT = """An underwriting agent made a wrong decision.
Application fields: {application}
Agent decided: {got}
Correct decision: {expected}

Write ONE short, GENERAL underwriting rule that would prevent this mistake.
Do NOT reference this specific application ID or any applicant details.
Reply with only the rule."""
```

Lessons are stored as a list of strings in `run_state.json`. Prepended to system prompt under `"Underwriting lessons learned from past decisions:"`.

**Gain measurement (honest protocol from RSI Book / CL-BENCH):**
```python
stateless_score, _ = evaluate(SEED_PROMPT, split="test")
stateful_score, _  = evaluate(with_lessons(SEED_PROMPT, lessons), split="test")
gain = stateful_score - stateless_score
print(f"gain: {gain:+.0%}")
```

---

## 9. Defenses

| Defense | Implementation |
|---------|---------------|
| Train/test split | Loop 4 only sees train failures; test IDs (APP-017–020) never appear in train set |
| Judge checksum | SHA-256 of `historical_decisions.json` recorded in `run_state.json` at run start; verified before any score counts |
| Prompt auditor | Separate `claude-sonnet-4-6` call after each accepted mutation; checks for hardcoded app IDs or specific decisions; verdict logged but never fed back to Loop 4 |

---

## 10. Cost Metering

**`billing/cost_meter.py`**

Wraps the Anthropic client. Tracks `input_tokens` and `output_tokens` per call, estimates cost using published pricing for `claude-sonnet-4-6`.

```python
BUDGET_USD = 2.00

class MeteredClient:
    def __init__(self, client: Anthropic):
        self._client = client
        self.spent = 0.0

    def create(self, **kwargs) -> Message:
        if self.spent >= BUDGET_USD:
            raise BudgetExhaustedError(f"Budget exhausted at ${self.spent:.4f}")
        response = self._client.messages.create(**kwargs)
        self.spent += self._estimate_cost(response.usage)
        return response
```

Receipt at end of run:
```
spend:              $0.18
train score:        100%
test score:         75%
gain (vs baseline): +50%
cost per point:     $0.0036
```

---

## 11. Entry Point

**`main.py`** — three modes:

```bash
# Single application through all 4 loops (good for debugging)
python -m src.main --mode single --app-id APP-001

# Full event loop: process all pending apps, trigger hill-climbing every 8
python -m src.main --mode event-loop [--once]

# Standalone improvement loop: run hill-climbing N rounds on train split
python -m src.main --mode improve --rounds 5
```

---

## 12. Dependencies

```toml
[project]
name = "underwriting_loop"
requires-python = ">=3.10"

dependencies = [
    "anthropic>=0.40.0",
    "langgraph>=0.2.0",
    "langchain-anthropic>=0.3.0",
    "pydantic>=2.0",
]
```

No LangSmith, no external databases. Everything runs locally.

---

## 13. What Is Explicitly Out of Scope

- Real insurance data or actual risk APIs
- Fine-tuning or weight-level modification
- A UI or web dashboard
- LangSmith deployment or webhook integration
- Multi-tenant or production deployment concerns
