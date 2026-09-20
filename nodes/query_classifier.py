"""
nodes/query_classifier.py
--------------------------
LangGraph node: classify_query

Classifies the user's question into PDF_ONLY, WEB_ONLY, or HYBRID using
structured output from the LLM.  Falls back to a keyword heuristic when
the LLM call fails (e.g. Ollama not running).
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from llm_provider import get_structured_llm, get_llm
from schemas import QueryClassification

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a query routing expert for a research assistant.

You must classify the user's question into exactly ONE of three routes:

PDF_ONLY  – The question is entirely about the content of the uploaded document(s).
            Examples: summarise the PDF, what does chapter 3 say, what dataset was used.

WEB_ONLY  – The question requires current/live information that cannot be in any PDF.
            Examples: latest OpenAI pricing, what happened in AI this week, current stock price.

HYBRID    – The question requires BOTH the uploaded document AND current web information.
            Examples: compare the method in my PDF with current research, is this technique still used.

Rules:
- If the user references "the document", "the PDF", "the paper", "this file", lean toward PDF_ONLY or HYBRID.
- If the user asks about "latest", "current", "recent", "today", "this week", lean toward WEB_ONLY or HYBRID.
- Always return a concise one-sentence reason.
- Respond in valid JSON format matching the schema with "route" and "reason" keys.
"""

CLASSIFIER_HUMAN = "Question: {question}"

# ── Heuristic fallback ────────────────────────────────────────────────────────

_WEB_KEYWORDS = {
    "latest", "current", "recent", "today", "this week", "this month",
    "now", "2024", "2025", "2026", "price", "pricing", "news", "update",
    "new release", "just released", "announced",
}
_PDF_KEYWORDS = {
    "document", "pdf", "paper", "file", "uploaded", "chapter",
    "section", "page", "figure", "table", "according to", "in the",
    "this paper", "the paper", "the document", "the file",
}


def _heuristic_classify(question: str, has_document: bool = True) -> QueryClassification:
    q_lower = question.lower()
    has_web = any(kw in q_lower for kw in _WEB_KEYWORDS)
    has_pdf = any(kw in q_lower for kw in _PDF_KEYWORDS)

    if not has_document:
        if has_pdf:
            route = "PDF_ONLY"
            reason = "Question references a document, but no PDF is currently indexed."
        else:
            route = "WEB_ONLY"
            reason = "No PDF document is indexed; searching the web."
    elif has_pdf and has_web:
        route = "HYBRID"
        reason = f"Detected document and web query parameters."
    elif has_web:
        route = "WEB_ONLY"
        reason = f"Detected web search keywords."
    else:
        route = "PDF_ONLY"
        reason = f"Defaulting to indexed PDF document context."

    return QueryClassification(
        route=route,
        reason=reason,
    )


# ── Main node ─────────────────────────────────────────────────────────────────

def classify_query(state: dict) -> dict:
    """
    LangGraph node.

    Reads  : state["question"], state.get("has_document", False)
    Writes : state["route"], state["route_reason"]
    """
    question = state.get("question", "")
    has_document = state.get("has_document", False)

    if not question.strip():
        return {"route": "PDF_ONLY", "route_reason": "Empty question — defaulting to PDF_ONLY."}

    q_lower = question.lower()
    has_pdf_kw = any(kw in q_lower for kw in _PDF_KEYWORDS)
    has_web_kw = any(kw in q_lower for kw in _WEB_KEYWORDS)

    # ── Fast-path 1: No document indexed and no document keywords -> 100% WEB_ONLY (<1ms)
    if not has_document and not has_pdf_kw:
        logger.info("classify_query (fast-path): route=WEB_ONLY (no document indexed)")
        return {
            "route": "WEB_ONLY",
            "route_reason": "No PDF document is indexed; searching the web directly.",
        }

    # ── Fast-path 2: Document indexed, explicit document keywords, no web keywords -> 100% PDF_ONLY (<1ms)
    if has_document and has_pdf_kw and not has_web_kw:
        logger.info("classify_query (fast-path): route=PDF_ONLY (document query)")
        return {
            "route": "PDF_ONLY",
            "route_reason": "Direct question about the active PDF document.",
        }

    # ── Try structured LLM output for ambiguous queries ──────────────────────
    try:
        doc_context_note = (
            "NOTE: An uploaded document is currently active and indexed."
            if has_document
            else "NOTE: NO document is uploaded right now. Choose WEB_ONLY unless the user specifically asks about an uploaded document/PDF."
        )
        structured_llm = get_structured_llm(QueryClassification)
        messages = [
            SystemMessage(content=f"{CLASSIFIER_SYSTEM}\n\n{doc_context_note}"),
            HumanMessage(content=CLASSIFIER_HUMAN.format(question=question)),
        ]
        result: QueryClassification = structured_llm.invoke(messages)

        route = result.route
        reason = result.reason

        # If no document is uploaded, safeguard against phantom PDF routing
        if not has_document:
            has_pdf_keywords = any(kw in question.lower() for kw in _PDF_KEYWORDS)
            if route == "HYBRID":
                route = "WEB_ONLY"
                reason += " (Adjusted to WEB_ONLY because no PDF is indexed)."
            elif route == "PDF_ONLY" and not has_pdf_keywords:
                route = "WEB_ONLY"
                reason = "General query routed to WEB_ONLY because no PDF is indexed."

        logger.info(
            "classify_query: route=%s | reason=%s", route, reason
        )
        return {"route": route, "route_reason": reason}

    except Exception as llm_err:
        logger.warning("classify_query LLM failed (%s); using heuristic.", llm_err)

    # ── Heuristic fallback ────────────────────────────────────────────────────
    fallback = _heuristic_classify(question, has_document=has_document)
    logger.info(
        "classify_query (heuristic): route=%s | reason=%s",
        fallback.route,
        fallback.reason,
    )
    return {"route": fallback.route, "route_reason": fallback.reason}
