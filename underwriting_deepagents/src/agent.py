from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from .billing import CostMeter
from .middleware import RubricMiddleware, UNDERWRITING_RUBRIC
from .tools.application_tool import lookup_application
from .tools.decision_tool import draft_decision
from .tools.risk_signal_tool import fetch_risk_signal
from .tools.rules_tool import retrieve_rules

SEED_PROMPT = """You are an insurance underwriting agent. Evaluate each application systematically.

For each application:
1. Call lookup_application to get the full application record
2. Call retrieve_rules with relevant risk factors (e.g. "flood", "claims", "credit", "coverage", "state")
3. Call fetch_risk_signal for additional data (credit, flood, property, claims)
4. Call draft_decision with your final decision, rationale, all applicable flags, and premium adjustment

Decisions: accept | modify | decline | refer
- accept: standard risk, approve as-is
- modify: approve with required conditions or endorsements
- decline: risk exceeds acceptable threshold
- refer: requires senior underwriter review

Always call draft_decision as your final action. Include all applicable flags."""

TOOLS = [lookup_application, retrieve_rules, fetch_risk_signal, draft_decision]


def build_agent(system_prompt: str, meter: CostMeter):
    model = ChatAnthropic(model="claude-sonnet-4-6")
    return create_react_agent(model=model, tools=TOOLS, state_modifier=system_prompt)


def build_verified_agent(system_prompt: str, meter: CostMeter) -> RubricMiddleware:
    agent = build_agent(system_prompt, meter)
    return RubricMiddleware(agent, rubric=UNDERWRITING_RUBRIC, max_retries=3)
