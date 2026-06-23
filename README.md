# Loop Engineering — Self-Improving Insurance Underwriting Agents

Two companion projects that implement the same four-level agentic loop for insurance underwriting — one using **LangGraph** (manual control), one using **LangChain's `create_react_agent` + a custom `RubricMiddleware`** (higher abstraction). Same domain, same data, different engineering tradeoffs.

---

## Architecture: Four Nested Loops

```
Loop 4 — Hill Climbing (reflect → mutate → evaluate → gate)
  └─ Loop 3 — Event Queue Poller (processes pending applications)
       └─ Loop 2 — Verification / Retry (grader-based retry on wrong decisions)
            └─ Loop 1 — ReAct Agent (tool-calling: lookup → rules → signals → decide)
```

| | `underwriting_loop` | `underwriting_deepagents` |
|---|---|---|
| Loop 1+2 implementation | Manual `StateGraph` (~350 lines) | `create_react_agent` + `RubricMiddleware` (~150 lines) |
| Loop 3 | `loop3_event.py` | `event_loop.py` |
| Loop 4 | `loop4_hill_climbing.py` | `hill_climbing.py` |
| Model | `claude-sonnet-4-6` | `claude-sonnet-4-6` |
| Tests (offline) | 50 | 24 |

---

## Dataset

20 insurance applications (16 train / 4 test) with 5 underwriting rules:

| Rule | Trigger | Required decision / flag |
|---|---|---|
| RULE-001 Flood | `flood_zone=true` + no rider | `modify` + `flood_rider_required` |
| RULE-002 Claims | `claim_history_5yr > 2` | `decline` or `refer` + `excess_claims` |
| RULE-003 Credit | `credit_score_tier=poor` | `decline` or surcharge ≥ 15% + `poor_credit` |
| RULE-004 Coverage | `coverage_amount > $1M` | `refer` + `high_coverage` |
| RULE-005 Geography | `state=FL` or `TX` | `windstorm_exclusion` flag (any decision) |

Ground truth lives in `src/data/historical_decisions.json`. A SHA-256 checksum is verified at runtime to prevent tampering.

---

## Project A — `underwriting_loop` (LangGraph)

### Setup

```bash
cd underwriting_loop
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run offline tests (no API key needed)

```bash
pytest tests/ -v
```

**50 tests across 8 modules — all fully mocked, no API calls:**

| Test file | What it covers |
|---|---|
| `test_tools.py` | 4 LangChain tools: lookup, rules, signals, decision |
| `test_cost_meter.py` | Budget tracking, `BudgetExhaustedError` kill switch |
| `test_judge.py` | Deterministic grader, train/test split, checksum |
| `test_loop1.py` | ReAct agent via LangGraph `StateGraph` |
| `test_loop2.py` | Retry wrapper: pass, retry-then-pass, exhaust, refer bypass |
| `test_loop3.py` | Event queue: load state, update state, process one app |
| `test_loop4.py` | Hill climbing: keep better, revert worse, discard cheating |
| `test_memory.py` | Lesson injection, distillation mock, gain measurement |

Run a specific module:

```bash
pytest tests/test_loop4.py -v
```

### Run modes (requires `ANTHROPIC_API_KEY`)

**Single application (debug / inspect):**

```bash
python -m src.main --mode single --app-id APP-001
```

Expected output for APP-001 (FL flood zone, no rider):
```
Decision:  modify
Flags:     ['flood_rider_required', 'windstorm_exclusion']
Rationale: Property is in a flood zone without a flood rider...
spend: $0.0049
```

**Process full queue once (16 apps → hill-climb after every 8):**

```bash
python -m src.main --mode event-loop --once
```

**Standalone hill-climbing (5 rounds on train split):**

```bash
python -m src.main --mode improve --rounds 5
```

Expected output:
```
seed score: 35–50%
round 1: 62% KEPT  (6 failures)
round 2: 75% KEPT  (4 failures)
round 3: 75% REVERTED (75% <= 75%)
...
test score: 75%  gain: +25%
spend: $1.23
```

### Key files

```
underwriting_loop/
├── src/
│   ├── loops/
│   │   ├── loop1_agent.py         # LangGraph StateGraph ReAct agent
│   │   ├── loop2_verification.py  # Grader retry + human queue for 'refer'
│   │   ├── loop3_event.py         # Queue poller, triggers hill-climb every 8
│   │   └── loop4_hill_climbing.py # reflect() → evaluate() → audit → gate
│   ├── judge/
│   │   ├── judge.py               # Deterministic grader (no LLM)
│   │   └── audit.py               # SHA-256 checksum + prompt cheating detector
│   ├── memory/lesson_memory.py    # Lesson distillation + with_lessons()
│   ├── billing/cost_meter.py      # Token cost tracking, $2 hard cap
│   └── main.py                    # CLI: --mode single|event-loop|improve
├── state/
│   ├── pending_applications.json  # Queue of app IDs to process
│   ├── run_state.json             # Processed/passed/failed/lessons
│   └── best_prompt.txt            # Current best system prompt (updated in-place)
└── tests/                         # 50 offline unit tests
```

---

## Project B — `underwriting_deepagents` (create_react_agent + RubricMiddleware)

### Setup

```bash
cd underwriting_deepagents
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run offline tests (no API key needed)

```bash
pytest tests/ -v
```

**24 tests across 6 modules — all fully mocked:**

| Test file | What it covers |
|---|---|
| `test_tools.py` | Same 4 tools as Project A |
| `test_judge.py` | Billing, checksum, judge, `evaluate()` with agent factory |
| `test_middleware.py` | `check_rubric()`: valid, missing flood flag, wrong claims decision; retry loop |
| `test_agent.py` | `build_agent()` returns runnable, `build_verified_agent()` wraps with middleware |
| `test_hill_climbing.py` | Keep better prompt, discard cheating prompt |
| `test_event_loop.py` | State update, process one app end-to-end (mocked agent) |

### Run modes (requires `ANTHROPIC_API_KEY`)

**Single application:**

```bash
python -m src.main --mode single --app-id APP-006
```

APP-006 (CA residential, clean profile) should return `accept` with 1 attempt (no rubric retry needed).

**Event loop:**

```bash
python -m src.main --mode event-loop --once
```

**Hill-climbing:**

```bash
python -m src.main --mode improve --rounds 5
```

### Key difference from Project A: RubricMiddleware

Project B enforces underwriting rules *structurally* — before hill-climbing even runs. If the agent returns a decision that violates a rubric rule, it retries up to 3 times with inline feedback:

```python
# 9 rubric rules checked on every response:
- decision_valid          (must be accept|modify|decline|refer)
- rationale_present       (non-empty rationale)
- flood_rider_required    (flood zone + no rider → flag required)
- flood_decision          (flood zone + no rider → not accept)
- excess_claims_flag      (>2 claims → excess_claims flag)
- excess_claims_decision  (>2 claims → decline or refer only)
- poor_credit             (poor tier + accept → needs ≥15% surcharge)
- high_coverage           (>$1M → must refer)
- windstorm_exclusion     (FL/TX → flag required)
```

This gives a higher floor accuracy before any hill-climbing.

### Key files

```
underwriting_deepagents/
├── src/
│   ├── agent.py          # create_react_agent + build_verified_agent()
│   ├── middleware.py     # RubricMiddleware class + UNDERWRITING_RUBRIC dict
│   ├── judge.py          # evaluate(agent_factory, prompt, split) — factory pattern
│   ├── audit.py          # SHA-256 checksum + audit_prompt()
│   ├── billing.py        # CostMeter (identical to Project A)
│   ├── memory.py         # with_lessons(), distill_lesson(), measure_gain()
│   ├── hill_climbing.py  # reflect() → evaluate() → audit → gate
│   ├── event_loop.py     # Queue poller (Loop 3)
│   └── main.py           # CLI: --mode single|event-loop|improve
├── state/
│   ├── pending_applications.json
│   ├── run_state.json
│   └── best_prompt.txt
└── tests/                # 24 offline unit tests
```

---

## Running both test suites at once

From the repo root:

```bash
# Project A
cd underwriting_loop && pytest tests/ -v && cd ..

# Project B
cd underwriting_deepagents && pytest tests/ -v && cd ..
```

Expected: **50 passed** + **24 passed**, no API calls, ~10s each.

---

## Budget & Safety

Both projects share the same safeguards:

- **Hard budget cap** — `$2.00` per run. Raises `BudgetExhaustedError` on the next API call once exceeded.
- **Judge checksum** — SHA-256 of `historical_decisions.json` is stored in `run_state.json` and verified at startup. Modification of ground truth breaks the run.
- **Prompt auditor** — Before accepting a hill-climbed prompt, `audit_prompt()` calls `claude-sonnet-4-6` to check for hardcoded app IDs or memorized decisions. Any `VERDICT: CHEATING` result discards the candidate.
- **Human queue** — Applications where the agent decides `refer` are written to `state/human_queue.json` and bypassed in the grader.

---

## Pricing reference (claude-sonnet-4-6)

| | Tokens | Cost |
|---|---|---|
| Input | per 1M | $3.00 |
| Output | per 1M | $15.00 |
| Single app (est.) | ~1,500 in / 300 out | ~$0.009 |
| Full 16-app eval | ~24k in / 5k out | ~$0.15 |
| 5-round improve | ~5 evals + reflections | ~$0.80–$1.50 |
