"""
nodes/pdf_retriever.py
----------------------
LangGraph node: retrieve_pdf

Wraps the existing LlamaIndex + ChromaDB pipeline.
Reads the query engine stored in Streamlit session_state (injected by the
UI layer before graph execution) and retrieves the top-k chunks, preserving
all metadata required for downstream citation building.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel used when the query engine is not yet initialised
_NO_ENGINE = object()


def retrieve_pdf(state: dict, query_engine: Any = None) -> dict:
    """
    LangGraph node.

    Parameters
    ----------
    state        : AgentState dict
    query_engine : LlamaIndex QueryEngine instance (injected by the graph runner)

    Returns
    -------
    Partial AgentState with ``pdf_documents`` populated.
    """
    question = state.get("question", "")

    if query_engine is None:
        logger.warning("retrieve_pdf: no query engine available — returning empty results")
        return {"pdf_documents": []}

    try:
        # Use retrieve() to get raw nodes with scores instead of query()
        # so we preserve all metadata without generating an answer here.
        retriever = query_engine.retriever
        nodes = retriever.retrieve(question)

        pdf_docs = []
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = node.metadata or {}

            pdf_docs.append(
                {
                    "content": node.get_content(),
                    "score": float(node_with_score.score or 0.0),
                    "source_type": "pdf",
                    # Citation fields
                    "filename": metadata.get("file_name", metadata.get("filename", "uploaded_document.pdf")),
                    "page_number": metadata.get("page_label", metadata.get("page_number", "?")),
                    "chunk_id": node.node_id,
                    "doc_id": metadata.get("doc_id", ""),
                }
            )

        logger.info("retrieve_pdf: retrieved %d chunks", len(pdf_docs))
        return {"pdf_documents": pdf_docs}

    except Exception as exc:
        logger.error("retrieve_pdf error: %s", exc, exc_info=True)
        return {"pdf_documents": []}
