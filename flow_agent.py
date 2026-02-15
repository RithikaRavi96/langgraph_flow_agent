from typing import TypedDict, Literal, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import json

# ---------------------------
# LLM
# ---------------------------
LLM = ChatOllama(model="llama3", temperature=0.2)

# ---------------------------
# State
# ---------------------------
class AgentState(TypedDict):
    user_input: str
    route: Literal["tool", "direct"]
    tool_name: Optional[Literal["calculator", "none"]]
    tool_input: Optional[str]
    tool_result: Optional[str]
    answer: str
    error: Optional[str]

# ---------------------------
# Tool (safe-ish calculator)
# ---------------------------
def calculator(expr: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expr):
        raise ValueError("Unsupported characters in expression.")
    return str(eval(expr, {"__builtins__": {}}, {}))

# ---------------------------
# Nodes
# ---------------------------
def route_node(state: AgentState) -> AgentState:
    prompt = (
        "You are a router.\n"
        "If the user asks to calculate something, convert units, or do arithmetic -> TOOL.\n"
        "Otherwise -> DIRECT.\n"
        f"User: {state['user_input']}\n"
        "Return only: TOOL or DIRECT."
    )
    decision = LLM.invoke(prompt).content.strip().upper()
    decision = decision.split()[0]  # avoids 'NO TOOL' bug
    state["route"] = "tool" if decision == "TOOL" else "direct"
    return state

def plan_tool_node(state: AgentState) -> AgentState:
    prompt = (
        "You are a planner.\n"
        "If the user asks for arithmetic, choose calculator and extract the expression.\n"
        "Otherwise choose none.\n"
        "Return ONLY valid JSON. No extra text.\n"
        'Schema: {"tool_name":"calculator"|"none","tool_input":"..."}\n'
        'Example: User:"12*7 + 5" -> {"tool_name":"calculator","tool_input":"12*7 + 5"}\n'
        'Example: User:"Explain tool calling" -> {"tool_name":"none","tool_input":""}\n'
        f'User:"{state["user_input"]}"'
    )
    raw = LLM.invoke(prompt).content.strip()

    try:
        plan_json = json.loads(raw)
        state["tool_name"] = plan_json.get("tool_name", "none")
        state["tool_input"] = plan_json.get("tool_input", "")
        state["error"] = None
    except Exception:
        state["tool_name"] = "none"
        state["tool_input"] = ""
        state["error"] = f"Failed to parse JSON plan: {raw}"

    return state

def run_tool_node(state: AgentState) -> AgentState:
    try:
        if state.get("tool_name") == "calculator":
            state["tool_result"] = calculator(state.get("tool_input") or "")
            state["error"] = None
        else:
            state["tool_result"] = None
            state["error"] = "No valid tool selected."
    except Exception as e:
        state["tool_result"] = None
        state["error"] = f"Tool failed: {e}"
    return state

def fallback_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "I can help — quick question so I don’t guess: "
        "what exact calculation or input should I use?"
    )
    return state

def direct_answer_node(state: AgentState) -> AgentState:
    prompt = f"Answer clearly and practically.\nUser: {state['user_input']}"
    state["answer"] = LLM.invoke(prompt).content.strip()
    return state

def final_answer_node(state: AgentState) -> AgentState:
    prompt = (
        "Use the tool result to answer the user concisely.\n"
        f"User: {state['user_input']}\n"
        f"Tool result: {state.get('tool_result')}\n"
    )
    state["answer"] = LLM.invoke(prompt).content.strip()
    return state

# ---------------------------
# Conditions
# ---------------------------
def route_condition(state: AgentState) -> str:
    return "tool_path" if state["route"] == "tool" else "direct_path"

def plan_condition(state: AgentState) -> str:
    return "use_tool" if state.get("tool_name") == "calculator" else "skip_tool"

def tool_success_condition(state: AgentState) -> str:
    return "fallback_path" if state.get("error") else "success_path"

# ---------------------------
# Build graph
# ---------------------------
graph = StateGraph(AgentState)

graph.add_node("route", route_node)
graph.add_node("plan_tool", plan_tool_node)
graph.add_node("run_tool", run_tool_node)
graph.add_node("fallback", fallback_node)
graph.add_node("direct_answer", direct_answer_node)
graph.add_node("final_answer", final_answer_node)

graph.set_entry_point("route")

# route -> plan_tool OR direct_answer
graph.add_conditional_edges("route", route_condition, {
    "tool_path": "plan_tool",
    "direct_path": "direct_answer",
})

# plan_tool -> run_tool OR direct_answer (if tool_name == none)
graph.add_conditional_edges("plan_tool", plan_condition, {
    "use_tool": "run_tool",
    "skip_tool": "direct_answer",
})

# run_tool -> final OR fallback
graph.add_conditional_edges("run_tool", tool_success_condition, {
    "success_path": "final_answer",
    "fallback_path": "fallback",
})

graph.add_edge("direct_answer", END)
graph.add_edge("final_answer", END)
graph.add_edge("fallback", END)

APP = graph.compile()

# ---------------------------
# CLI
# ---------------------------
def main():
    print("LangGraph Flow Agent (no RAG). Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        state: AgentState = {
            "user_input": user,
            "route": "direct",
            "tool_name": None,
            "tool_input": None,
            "tool_result": None,
            "answer": "",
            "error": None,
        }

        out = APP.invoke(state)

        print("\n--- TRACE ---")
        print("route:", out.get("route"))
        print("tool_name:", out.get("tool_name"))
        print("tool_input:", out.get("tool_input"))
        print("tool_result:", out.get("tool_result"))
        print("error:", out.get("error"))
        print("-------------\n")

        print("Assistant:", out["answer"], "\n")

if __name__ == "__main__":
    main()