"""
nodes/retry_searcher.py
-----------------------
LangGraph node: retry_search

When evidence is insufficient, this node:
  1. Uses the LLM to reformulate the query based on missing_information.
  2. Performs a new Tavily web search with the improved query.
  3. Appends new results to existing web_documents without discarding them.

MAX_RETRIES = 2  — prevents infinite loops.
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage

from llm_provider import get_structured_llm, get_llm
from schemas import RetryQuery
from nodes.web_searcher import search_web as _search_web

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

REFORMULATE_SYSTEM = """You are a search query optimisation expert.

A previous search did not return enough information to answer the user's question.
Your job is to reformulate the query to be more specific, targeted, and likely
to find the missing information.

Techniques:
- Add domain-specific terminology.
- Break a vague question into its core information need.
- Add temporal qualifiers if current information is needed (e.g. "2025").
- Use alternative phrasings or synonyms.
- Do NOT simply repeat the original query.
"""

REFORMULATE_HUMAN = """Original question: {question}

Previous search query: {previous_query}

Missing information identified: {missing_information}

Evidence grader feedback: {grade_reason}

Write a better search query that directly targets the missing information."""


def retry_search(state: dict) -> dict:
    """
    LangGraph node.

    Reads  : state["question"], state["search_query"],
             state["missing_information"], state["grade_reason"],
             state["retry_count"], state["web_documents"]
    Writes : state["web_documents"] (appended), state["search_query"],
             state["retry_count"]
    """
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        logger.info("retry_search: MAX_RETRIES (%d) reached — skipping retry.", MAX_RETRIES)
        return {"retry_count": retry_count}

    question = state.get("question", "")
    previous_query = state.get("search_query") or question
    missing_info = state.get("missing_information", "")
    grade_reason = state.get("grade_reason", "")

    # ── Reformulate the query ─────────────────────────────────────────────────
    new_query = previous_query  # default: reuse if LLM fails

    try:
        structured_llm = get_structured_llm(RetryQuery)
        messages = [
            SystemMessage(content=REFORMULATE_SYSTEM),
            HumanMessage(
                content=REFORMULATE_HUMAN.format(
                    question=question,
                    previous_query=previous_query,
                    missing_information=missing_info,
                    grade_reason=grade_reason,
                )
            ),
        ]
        result: RetryQuery = structured_llm.invoke(messages)
        new_query = result.reformulated_query
        logger.info(
            "retry_search: reformulated query: %r (reason: %s)",
            new_query,
            result.reason,
        )
    except Exception as err:
        logger.warning("retry_search reformulation failed (%s); using fallback query.", err)
        # Simple fallback: append missing info keywords to original
        if missing_info:
            new_query = f"{question} {missing_info}"

    # ── Execute new search ────────────────────────────────────────────────────
    search_result = _search_web(state, query=new_query)
    new_web_docs = search_result.get("web_documents", [])

    # Append new results to existing (don't discard prior evidence)
    existing_web_docs = state.get("web_documents") or []
    combined_web_docs = existing_web_docs + new_web_docs

    logger.info(
        "retry_search: added %d new web results (total: %d)",
        len(new_web_docs),
        len(combined_web_docs),
    )

    return {
        "web_documents": combined_web_docs,
        "search_query": new_query,
        "retry_count": retry_count + 1,
    }
