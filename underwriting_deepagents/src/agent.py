from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.prebuilt import create_react_agent

from .billing import BudgetExhaustedError, CostMeter
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


class _MeterCallback(BaseCallbackHandler):
    """Records Anthropic token usage into a CostMeter after each LLM call."""

    def __init__(self, meter: CostMeter):
        self._meter = meter

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    try:
                        self._meter.record(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )
                    except BudgetExhaustedError:
                        pass  # LangChain swallows callback exceptions anyway; caller checks meter.check()
                    return


def build_agent(system_prompt: str, meter: CostMeter):
    model = ChatAnthropic(model="claude-sonnet-4-6", callbacks=[_MeterCallback(meter)])
    return create_react_agent(model=model, tools=TOOLS, prompt=system_prompt)


def build_verified_agent(system_prompt: str, meter: CostMeter) -> RubricMiddleware:
    agent = build_agent(system_prompt, meter)
    return RubricMiddleware(agent, rubric=UNDERWRITING_RUBRIC, max_retries=3, meter=meter)
