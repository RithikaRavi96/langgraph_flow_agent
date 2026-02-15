# LangGraph Flow Agent (Ollama) — Routing + Tool Use + Fallback

This repo is a minimal LangGraph demo to showcase **agentic orchestration patterns**:
A minimal LangGraph-based agent demonstrating structured routing, tool invocation, fallback handling, and traceable execution flow — built using LangGraph + Ollama (Llama3).

This project focuses on agent orchestration logic, not RAG or retrieval systems.

🚀 What This Demonstrates

✅ Router node (LLM decides: tool vs direct response)

✅ Tool planning logic

✅ Tool execution (safe calculator example)

✅ Validation & fallback handling

✅ Conditional graph routing

✅ Execution trace printing

✅ Clean state-based agent architecture

This is designed to show understanding of:

Agent state machines

Tool-calling workflows

Conditional routing

Fallback logic

Structured orchestration

Traceable execution

🏗 Architecture Overview

User Input
⬇
Router (LLM decides: TOOL or DIRECT)
⬇
If TOOL → Plan Tool → Run Tool → Validate → Final Answer
⬇
If DIRECT → Direct LLM Answer
⬇
Fallback if tool fails

Built using StateGraph from LangGraph.

📂 Tech Stack

Python

LangGraph

LangChain

Ollama (Llama3 local model)

TypedDict state management

🛠 Setup Instructions
1️⃣ Create Virtual Environment
python -m venv .venv

Activate:

Windows

.venv\Scripts\activate

Mac/Linux

source .venv/bin/activate
2️⃣ Install Dependencies
pip install langgraph langchain langchain-ollama
3️⃣ Install & Run Ollama

Download Ollama:

👉 https://ollama.com

Pull Llama3 model:

ollama pull llama3

Make sure Ollama is running locally.

4️⃣ Run the Agent
python agent.py

Example:

You: 12 * 5 + 3

You’ll see:

--- TRACE ---
route: tool
tool_name: calculator
tool_input: 12 * 5 + 3
tool_result: 63
error: None
-------------
Assistant: The result is 63.
🔎 Why This Project Matters

This project shows:

Thinking in LLM-native workflows

Understanding of tool calling vs direct generation

State-based agent orchestration

Handling ambiguity safely with fallback

Clean, production-style separation of concerns

This is not a simple RAG chatbot — it demonstrates structured agent logic flow.

📌 Future Improvements

Replace calculator with multi-tool setup

Add LangGraph multi-step agent loops

Add tool confidence scoring

Add structured output parsing

Add memory module

Deploy via FastAPI

👩‍💻 Author

Rithika Ravichandran
Machine Learning Engineer | Applied LLM Systems | Agentic Workflows
