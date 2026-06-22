# src/loops/loop1_agent.py
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..billing.cost_meter import CostMeter
from ..tools.application_tool import lookup_application
from ..tools.decision_tool import draft_decision
from ..tools.risk_signal_tool import fetch_risk_signal
from ..tools.rules_tool import retrieve_rules

SEED_PROMPT = """You are an insurance underwriting agent. Evaluate each application systematically.

For each application:
1. Call lookup_application to get the full application record (e.g. 'APP-001')
2. Call retrieve_rules with relevant risk factors — try: flood, claims, credit, coverage, state, business
3. Call fetch_risk_signal for additional data — try: credit, flood, property, claims
4. Call draft_decision with your final decision, rationale, all applicable flags, and premium adjustment

Decisions: accept | modify | decline | refer
- accept: standard risk, approve as-is
- modify: approve with required conditions or endorsements
- decline: risk exceeds acceptable threshold
- refer: requires senior underwriter review

Always call draft_decision as your final action. Include all applicable flags."""

TOOLS = [lookup_application, retrieve_rules, fetch_risk_signal, draft_decision]
MAX_STEPS = 8

_TOOL_MAP = {t.name: t for t in TOOLS}


class UnderwritingState(TypedDict):
    application_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    decision: str
    rationale: str
    flags: list[str]
    recommended_premium_adj: float
    lessons: list[str]
    spend: float
    _steps: int


def _dispatch_tools(tool_calls: list) -> tuple[list[ToolMessage], dict]:
    results = []
    updates = {}
    for tc in tool_calls:
        tool_fn = _TOOL_MAP.get(tc["name"])
        if tool_fn is None:
            results.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))
            continue
        output = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        if tc["name"] == "draft_decision":
            updates = {
                "decision": tc["args"]["decision"],
                "rationale": tc["args"]["rationale"],
                "flags": tc["args"]["flags"],
                "recommended_premium_adj": tc["args"]["recommended_premium_adj"],
            }
    return results, updates


def build_loop1(system_prompt: str, meter: CostMeter):
    model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(TOOLS)

    def agent_node(state: UnderwritingState) -> dict:
        prompt = system_prompt
        if state.get("lessons"):
            prompt += "\n\nUnderwriting lessons:\n" + "\n".join(f"- {l}" for l in state["lessons"])
        msgs = [SystemMessage(content=prompt)] + list(state["messages"])
        response = model.invoke(msgs)
        meter.record_from_message(response)
        return {
            "messages": [response],
            "_steps": state.get("_steps", 0) + 1,
            "spend": meter.spent,
        }

    def tools_node(state: UnderwritingState) -> dict:
        last = state["messages"][-1]
        tool_results, updates = _dispatch_tools(last.tool_calls)
        return {"messages": tool_results, **updates}

    def should_continue(state: UnderwritingState) -> Literal["tools", "__end__"]:
        if state.get("_steps", 0) >= MAX_STEPS:
            return END
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(UnderwritingState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run_loop1(
    app_id: str,
    system_prompt: str,
    meter: CostMeter,
    lessons: list[str] = None,
    extra_messages: list = None,
) -> dict:
    graph = build_loop1(system_prompt, meter)
    init_msgs = [
        HumanMessage(
            content=(
                f"Evaluate insurance application {app_id}. "
                "Use all available tools to assess risk and call draft_decision with your final decision."
            )
        )
    ]
    if extra_messages:
        init_msgs.extend(extra_messages)
    initial = {
        "application_id": app_id,
        "messages": init_msgs,
        "decision": "ran_out_of_steps",
        "rationale": "",
        "flags": [],
        "recommended_premium_adj": 0.0,
        "lessons": lessons or [],
        "spend": meter.spent,
        "_steps": 0,
    }
    return graph.invoke(initial)
