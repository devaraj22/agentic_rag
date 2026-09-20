"""
nodes/web_searcher.py
---------------------
LangGraph node: search_web

Uses Tavily to perform live web searches.  The API key is read from the
environment variable TAVILY_API_KEY — never hard-coded.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _is_valid_tavily_key(key: Optional[str]) -> bool:
    if not key or not key.strip():
        return False
    if key.strip().startswith("tvly-your-") or "your-key-here" in key:
        return False
    return True


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Fallback search using DuckDuckGo (no API key required)."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        docs = []
        for r in results:
            content = r.get("body") or r.get("snippet") or ""
            if content:
                docs.append({
                    "content": content,
                    "title": r.get("title") or "Web Source",
                    "url": r.get("href") or r.get("link") or "",
                    "score": 0.85,
                    "source_type": "web",
                })
        logger.info("search_web (DuckDuckGo fallback): retrieved %d results for query: %r", len(docs), query)
        return docs
    except Exception as exc:
        logger.warning("DuckDuckGo fallback search failed: %s", exc)
        return []


def _get_tavily_client():
    """Lazy-import and construct TavilySearch client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not _is_valid_tavily_key(api_key):
        raise EnvironmentError(
            "TAVILY_API_KEY is not set or still contains placeholder."
        )

    try:
        from langchain_tavily import TavilySearch
        return TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_raw_content=False,
        )
    except ImportError:
        from langchain_community.tools.tavily_search import TavilySearchResults
        os.environ["TAVILY_API_KEY"] = api_key
        return TavilySearchResults(max_results=5)


def search_web(state: dict, query: Optional[str] = None) -> dict:
    """
    LangGraph node.

    Parameters
    ----------
    state : AgentState dict
    query : Optional override query (used in retry_search with a reformulated query)

    Returns
    -------
    Partial AgentState with ``web_documents`` populated.
    """
    search_query = query or state.get("search_query") or state.get("question", "")

    if not search_query.strip():
        logger.warning("search_web: empty query — skipping")
        return {"web_documents": []}

    # 1. Try Tavily if a valid key is provided
    try:
        tool = _get_tavily_client()
        raw_results = tool.invoke(search_query)

        # Handle TavilySearch dictionary output vs legacy list output
        items = []
        if isinstance(raw_results, dict):
            if "error" in raw_results:
                logger.warning("Tavily returned error: %s; falling back to DuckDuckGo", raw_results["error"])
                raise RuntimeError(str(raw_results["error"]))
            items = raw_results.get("results", [])
        elif isinstance(raw_results, list):
            items = raw_results

        web_docs = []
        for item in items:
            if isinstance(item, dict):
                content = item.get("content") or item.get("snippet") or item.get("body") or ""
                if content:
                    web_docs.append(
                        {
                            "content": content,
                            "title": item.get("title", "Web Source"),
                            "url": item.get("url") or item.get("href") or "",
                            "score": float(item.get("score", 0.8)),
                            "source_type": "web",
                        }
                    )
            elif isinstance(item, str) and item.strip():
                web_docs.append(
                    {
                        "content": item,
                        "title": "Web Source",
                        "url": "",
                        "score": 0.5,
                        "source_type": "web",
                    }
                )

        if web_docs:
            logger.info("search_web (Tavily): retrieved %d web results for query: %r", len(web_docs), search_query)
            return {"web_documents": web_docs, "search_query": search_query}
        else:
            logger.info("Tavily returned 0 results; falling back to DuckDuckGo")
            ddg_docs = _search_duckduckgo(search_query)
            return {"web_documents": ddg_docs, "search_query": search_query}

    except Exception as exc:
        logger.info("Tavily search unavailable (%s); using DuckDuckGo fallback", exc)
        ddg_docs = _search_duckduckgo(search_query)
        return {"web_documents": ddg_docs, "search_query": search_query}
