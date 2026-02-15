from __future__ import annotations

from typing import TypedDict, Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from prompts import SYSTEM_PROMPT, ROUTER_PROMPT
import tools


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    messages: List[Any]
    route: Optional[str]
    retrieved: Optional[List[Dict[str, str]]]
    tool_error: Optional[str]
    attempts: int


# -----------------------------
# Structured outputs
# -----------------------------
class RouteDecision(BaseModel):
    action: Literal["RETRIEVE", "TIME", "CHECK", "ANSWER"] = Field(..., description="Next action the agent should take.")


class FinalAnswer(BaseModel):
    answer: str = Field(..., description="Final user-facing answer.")
    used_tools: List[str] = Field(default_factory=list, description="Which tools were used.")
    notes: str = Field(default="", description="Short note about reliability/assumptions.")


# -----------------------------
# LLM
# -----------------------------
llm = ChatOllama(model="llama3", temperature=0.2)


def _last_user_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# -----------------------------
# Nodes
# -----------------------------
def router_node(state: AgentState) -> AgentState:
    user_text = _last_user_text(state)
    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{ROUTER_PROMPT}\n\nUser query:\n{user_text}")
    ]
    decision = llm.with_structured_output(RouteDecision).invoke(prompt)
    state["route"] = decision.action
    state["messages"].append(AIMessage(content=f"[router] action={decision.action}"))
    return state


def retrieve_node(state: AgentState) -> AgentState:
    q = _last_user_text(state)
    state["retrieved"] = tools.search_policy_snippets(q)
    state["messages"].append(AIMessage(content=f"[tool] retrieved {len(state['retrieved'])} snippets"))
    return state


def time_node(state: AgentState) -> AgentState:
    now = tools.get_time()
    state["messages"].append(AIMessage(content=f"[tool] current_time={now}"))
    return state


def check_node(state: AgentState) -> AgentState:
    # Demonstrate a tool that can fail + fallback
    try:
        msg = tools.flaky_dependency_check()
        state["tool_error"] = None
        state["messages"].append(AIMessage(content=f"[tool] check={msg}"))
    except Exception as e:
        state["tool_error"] = str(e)
        state["messages"].append(AIMessage(content=f"[tool] check_error={state['tool_error']}"))
    return state


def answer_node(state: AgentState) -> AgentState:
    user_text = _last_user_text(state)
    retrieved = state.get("retrieved") or []
    tool_error = state.get("tool_error")

    context = ""
    if retrieved:
        context_lines = [f"- {r['title']}: {r['text']}" for r in retrieved]
        context = "Retrieved context:\n" + "\n".join(context_lines)

    if tool_error:
        context += f"\n\nTool error observed: {tool_error}"

    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""User request:
{user_text}

{context}

Write the best possible answer. If uncertainty exists, say it and propose a safe next step.""")
    ]

    resp = llm.invoke(prompt)
    # minimal "used tools" inference from messages
    used_tools = []
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.content.startswith("[tool]"):
            if "retrieved" in m.content:
                used_tools.append("search_policy_snippets")
            if "current_time" in m.content:
                used_tools.append("get_time")
            if "check=" in m.content or "check_error" in m.content:
                used_tools.append("flaky_dependency_check")

    final = FinalAnswer(
        answer=resp.content,
        used_tools=sorted(list(set(used_tools))),
        notes=("A tool failed; I used fallback reasoning." if tool_error else "Answered with available context/tools.")
    )
    state["messages"].append(AIMessage(content=f"[final]\n{final.model_dump_json(indent=2)}"))
    return state


def fallback_node(state: AgentState) -> AgentState:
    # Simple fallback: if tool failed, do retrieval then answer
    state["messages"].append(AIMessage(content="[fallback] tool failed, switching to RETRIEVE then ANSWER"))
    state["route"] = "RETRIEVE"
    return state


# -----------------------------
# Routing logic
# -----------------------------
def route_after_router(state: AgentState) -> str:
    return state["route"] or "ANSWER"


def route_after_check(state: AgentState) -> str:
    # if check failed and we haven't tried too many times, fallback
    if state.get("tool_error") and state["attempts"] < 1:
        return "FALLBACK"
    return "ANSWER"


# -----------------------------
# Build graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("time", time_node)
graph.add_node("check", check_node)
graph.add_node("fallback", fallback_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    route_after_router,
    {
        "RETRIEVE": "retrieve",
        "TIME": "time",
        "CHECK": "check",
        "ANSWER": "answer",
    },
)

# after retrieve/time -> answer
graph.add_edge("retrieve", "answer")
graph.add_edge("time", "answer")

# after check -> either fallback or answer
graph.add_conditional_edges(
    "check",
    route_after_check,
    {
        "FALLBACK": "fallback",
        "ANSWER": "answer",
    },
)

# fallback goes to retrieve then answer
graph.add_edge("fallback", "retrieve")

graph.add_edge("answer", END)

app = graph.compile()


# -----------------------------
# CLI runner
# -----------------------------
def run_cli():
    print("LangGraph demo (Ollama llama3). Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        state: AgentState = {
            "messages": [HumanMessage(content=user)],
            "route": None,
            "retrieved": None,
            "tool_error": None,
            "attempts": 0,
        }

        # run once
        out = app.invoke(state)

        # Print the final JSON block nicely
        final_blocks = [m.content for m in out["messages"] if isinstance(m, AIMessage) and m.content.startswith("[final]")]
        if final_blocks:
            print("\nAssistant:")
            print(final_blocks[-1].replace("[final]\n", ""))
            print("")
        else:
            print("\nAssistant: (no final output)\n")


if __name__ == "__main__":
    run_cli()