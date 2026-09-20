"""
agent_graph.py
--------------
Constructs and compiles the LangGraph Agentic Research Assistant graph.

Graph topology
--------------
START
  → classify_query
  → [route: PDF_ONLY]  retrieve_pdf  → merge_evidence
  → [route: WEB_ONLY]  search_web    → merge_evidence
  → [route: HYBRID]    retrieve_pdf + search_web → merge_evidence
  → grade_evidence
  → [sufficient=True]  generate_answer
  → [sufficient=False, retries < MAX] retry_search → merge_evidence → grade_evidence
  → [sufficient=False, retries >= MAX] generate_answer   (with caveat)
  → build_citations
  → END

Each node is a pure function that receives and returns a partial AgentState dict.
The query engine (LlamaIndex) is injected via closure to avoid global state.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END

from agent_state import AgentState
from nodes.query_classifier import classify_query
from nodes.pdf_retriever import retrieve_pdf as _retrieve_pdf_raw
from nodes.web_searcher import search_web as _search_web_raw
from nodes.evidence_merger import merge_evidence
from nodes.evidence_grader import grade_evidence
from nodes.answer_generator import generate_answer
from nodes.retry_searcher import retry_search as _retry_search_raw, MAX_RETRIES
from nodes.citation_builder import build_citations

logger = logging.getLogger(__name__)


# ── Routing functions (conditional edges) ────────────────────────────────────

def route_after_classify(state: AgentState) -> Literal["retrieve_pdf", "search_web", "retrieve_pdf_and_web"]:
    route = state.get("route", "PDF_ONLY")
    if route == "PDF_ONLY":
        return "retrieve_pdf"
    elif route == "WEB_ONLY":
        return "search_web"
    else:  # HYBRID
        return "retrieve_pdf_and_web"


def route_after_grading(state: AgentState) -> Literal["generate_answer", "retry_search"]:
    sufficient = state.get("evidence_sufficient", False)
    retry_count = state.get("retry_count", 0)

    if sufficient:
        logger.debug("route_after_grading → generate_answer (sufficient)")
        return "generate_answer"
    elif retry_count < MAX_RETRIES:
        logger.debug(
            "route_after_grading → retry_search (retry %d/%d)", retry_count + 1, MAX_RETRIES
        )
        return "retry_search"
    else:
        logger.debug(
            "route_after_grading → generate_answer (max retries reached, insufficient evidence)"
        )
        return "generate_answer"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(query_engine: Any = None) -> Any:
    """
    Build and compile the LangGraph.

    Parameters
    ----------
    query_engine : LlamaIndex QueryEngine
        The existing PDF query engine from the Streamlit session.
        Injected via closure so nodes stay stateless functions.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke.
    """

    # ── Bind query_engine into retrieve_pdf ──────────────────────────────────
    def retrieve_pdf(state: AgentState) -> AgentState:
        return _retrieve_pdf_raw(state, query_engine=query_engine)

    # ── Hybrid node: PDF + Web in sequence ───────────────────────────────────
    def retrieve_pdf_and_web(state: AgentState) -> AgentState:
        pdf_result = _retrieve_pdf_raw(state, query_engine=query_engine)
        web_result = _search_web_raw({**state, **pdf_result})
        return {**pdf_result, **web_result}

    # ── Retry wrapper: re-merge after new web docs fetched ───────────────────
    def retry_search_then_merge(state: AgentState) -> AgentState:
        retry_result = _retry_search_raw(state)
        # Re-merge with existing PDF docs
        merged_state = {**state, **retry_result}
        merge_result = merge_evidence(merged_state)
        return {**retry_result, **merge_result}

    # ── Build the StateGraph ──────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_query",       classify_query)
    graph.add_node("retrieve_pdf",         retrieve_pdf)
    graph.add_node("search_web",           _search_web_raw)
    graph.add_node("retrieve_pdf_and_web", retrieve_pdf_and_web)
    graph.add_node("merge_evidence",       merge_evidence)
    graph.add_node("grade_evidence",       grade_evidence)
    graph.add_node("generate_answer",      generate_answer)
    graph.add_node("retry_search",         retry_search_then_merge)
    graph.add_node("build_citations",      build_citations)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Entry
    graph.add_edge(START, "classify_query")

    # Classify → route to appropriate retriever
    graph.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "retrieve_pdf":         "retrieve_pdf",
            "search_web":           "search_web",
            "retrieve_pdf_and_web": "retrieve_pdf_and_web",
        },
    )

    # All retrieval paths converge on merge_evidence
    graph.add_edge("retrieve_pdf",         "merge_evidence")
    graph.add_edge("search_web",           "merge_evidence")
    graph.add_edge("retrieve_pdf_and_web", "merge_evidence")

    # merge → grade
    graph.add_edge("merge_evidence", "grade_evidence")

    # grade → generate or retry
    graph.add_conditional_edges(
        "grade_evidence",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "retry_search":    "retry_search",
        },
    )

    # retry (already merges internally) → grade again
    graph.add_edge("retry_search", "grade_evidence")

    # generate → citations
    graph.add_edge("generate_answer", "build_citations")

    # citations → END
    graph.add_edge("build_citations", END)

    return graph.compile()
