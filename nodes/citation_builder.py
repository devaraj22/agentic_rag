"""
nodes/citation_builder.py
--------------------------
LangGraph node: build_citations

Constructs structured citation objects from the retrieved evidence.
Also assembles the final_response payload that the UI layer consumes.

PDF citation format : [PDF: filename, p.X]
Web citation format : [Web: Title](URL)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _make_pdf_citation(doc: dict, index: int) -> dict:
    filename = doc.get("filename", "document.pdf")
    page = doc.get("page_number", "?")
    chunk_id = doc.get("chunk_id", "")
    score = doc.get("score", 0.0)

    return {
        "index": index,
        "type": "pdf",
        "label": f"[PDF: {filename}, p.{page}]",
        "filename": filename,
        "page": page,
        "chunk_id": chunk_id,
        "relevance_score": round(score, 3),
    }


def _make_web_citation(doc: dict, index: int) -> dict:
    title = doc.get("title") or "Web Source"
    url = doc.get("url", "")
    score = doc.get("score", 0.0)
    error = doc.get("error", False)

    label = f"[Web: {title}]({url})" if url else f"[Web: {title}]"
    if error:
        label = f"[Web: Configuration Error]"

    return {
        "index": index,
        "type": "web",
        "label": label,
        "title": title,
        "url": url,
        "relevance_score": round(score, 3),
        "error": error,
    }


def build_citations(state: dict) -> dict:
    """
    LangGraph node.

    Reads  : state["pdf_documents"], state["web_documents"],
             state["answer"], state["question"], state["route"],
             state["evidence_score"], state["evidence_sufficient"],
             state["retry_count"]
    Writes : state["citations"], state["final_response"]
    """
    pdf_docs = state.get("pdf_documents") or []
    web_docs = state.get("web_documents") or []
    answer = state.get("answer", "")
    question = state.get("question", "")
    route = state.get("route", "")
    evidence_score = state.get("evidence_score", 0.0)
    evidence_sufficient = state.get("evidence_sufficient", False)
    retry_count = state.get("retry_count", 0)

    citations = []
    idx = 1

    # PDF citations — deduplicate by (filename, page)
    seen_pdf = set()
    for doc in pdf_docs:
        key = (doc.get("filename", ""), doc.get("page_number", ""))
        if key not in seen_pdf:
            citations.append(_make_pdf_citation(doc, idx))
            seen_pdf.add(key)
            idx += 1

    # Web citations — deduplicate by URL
    seen_web: set = set()
    for doc in web_docs:
        url = doc.get("url", "")
        url_key = url or doc.get("title", f"web_{idx}")
        if url_key not in seen_web and not doc.get("error"):
            citations.append(_make_web_citation(doc, idx))
            seen_web.add(url_key)
            idx += 1

    logger.info(
        "build_citations: %d PDF + %d web citations",
        len([c for c in citations if c["type"] == "pdf"]),
        len([c for c in citations if c["type"] == "web"]),
    )

    # ── Assemble final_response ───────────────────────────────────────────────
    final_response = {
        "question": question,
        "answer": answer,
        "citations": citations,
        "metadata": {
            "route": route,
            "evidence_score": round(evidence_score, 3),
            "evidence_sufficient": evidence_sufficient,
            "retry_count": retry_count,
            "pdf_sources_count": len(seen_pdf),
            "web_sources_count": len(seen_web),
        },
    }

    return {"citations": citations, "final_response": final_response}
