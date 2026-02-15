from __future__ import annotations
from datetime import datetime
from typing import List, Dict
import random

# Simple "tools" to show tool calling + possible failures

def get_time() -> str:
    """Return current time in ISO format."""
    return datetime.now().isoformat(timespec="seconds")

def search_policy_snippets(query: str) -> List[Dict[str, str]]:
    """
    Fake mini-retrieval tool (simulates RAG retrieval).
    Intentionally small, but looks realistic.
    """
    corpus = [
        {"title": "Risk Control: Verification", "text": "Risk controls must be verified and documented with traceable evidence."},
        {"title": "Risk Control: Usability", "text": "User-facing risk controls should be validated with representative users when applicable."},
        {"title": "Risk Control: Residual Risk", "text": "Residual risks must be evaluated and communicated when they remain unacceptable."},
        {"title": "AI Safety", "text": "For LLM outputs, include guardrails: citation of sources, uncertainty handling, and escalation paths."},
    ]
    q = query.lower()
    results = [d for d in corpus if any(w in (d["title"] + " " + d["text"]).lower() for w in q.split())]
    return results[:3] if results else corpus[:2]

def flaky_dependency_check() -> str:
    """
    Tool that sometimes fails to demonstrate fallback routing.
    """
    if random.random() < 0.35:
        raise RuntimeError("Simulated tool failure: dependency service unavailable")
    return "All dependency checks passed."