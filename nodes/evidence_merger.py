"""
nodes/evidence_merger.py
------------------------
LangGraph node: merge_evidence

Merges PDF chunks and web results into a single ordered evidence list while
preserving source provenance.  Sources are never silently mixed — each
document carries a ``source_type`` field ("pdf" or "web").
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def merge_evidence(state: dict) -> dict:
    """
    LangGraph node.

    Reads  : state["pdf_documents"], state["web_documents"]
    Writes : state["evidence"]
    """
    pdf_docs = state.get("pdf_documents") or []
    web_docs = state.get("web_documents") or []

    # Sort each group by descending relevance score, then interleave:
    # PDF first (highest scored), then web.  This ordering ensures the
    # grader sees the most relevant content from each source first.
    pdf_sorted = sorted(pdf_docs, key=lambda d: d.get("score", 0.0), reverse=True)
    web_sorted = sorted(web_docs, key=lambda d: d.get("score", 0.0), reverse=True)

    merged = pdf_sorted + web_sorted

    logger.info(
        "merge_evidence: %d PDF + %d web = %d total evidence items",
        len(pdf_sorted),
        len(web_sorted),
        len(merged),
    )

    return {"evidence": merged}
