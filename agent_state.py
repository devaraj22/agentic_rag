"""
agent_state.py
--------------
Shared LangGraph state object for the Agentic Hybrid RAG Research Assistant.
Every node reads from and writes to this TypedDict.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    question: str                   # Raw user question
    chat_history: list              # Prior messages for multi-turn context
    has_document: bool              # Whether a PDF document is currently loaded
    fast_mode: bool                 # Speed optimization mode

    # ── Routing ──────────────────────────────────────────────────────────────
    route: str                      # "PDF_ONLY" | "WEB_ONLY" | "HYBRID"
    route_reason: str               # LLM explanation for the routing decision

    # ── Retrieved Evidence ───────────────────────────────────────────────────
    pdf_documents: list             # Chunks from ChromaDB / LlamaIndex
    web_documents: list             # Results from Tavily

    # ── Grading ──────────────────────────────────────────────────────────────
    evidence: list                  # Merged, deduplicated evidence list
    evidence_score: float           # 0.0 – 1.0
    evidence_sufficient: bool       # Gate for generate vs retry
    grade_reason: str               # Explanation from grader LLM
    missing_information: str        # What gap was identified

    # ── Retry Control ────────────────────────────────────────────────────────
    retry_count: int                # Incremented on each retry
    search_query: str               # Reformulated query used in retry

    # ── Output ───────────────────────────────────────────────────────────────
    answer: str                     # Raw generated answer text
    citations: list                 # Structured citation objects
    final_response: dict            # Payload returned to UI
