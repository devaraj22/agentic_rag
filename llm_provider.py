"""
llm_provider.py
---------------
Single place to initialise the LLM used by every LangGraph node.

The existing app uses Ollama (llama3:8b-instruct-q4_0).  We keep that
exact model for all agentic nodes so there is no change to the local
inference setup.

For structured outputs we use langchain-ollama's ChatOllama with
.with_structured_output(), which calls the model via tool-calling /
JSON-mode depending on what the version supports.
"""

import os
from functools import lru_cache

from langchain_ollama import ChatOllama


# ── Constants ────────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_0")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "360"))


# ── Factory ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOllama:
    """Return a cached ChatOllama instance with warm keep-alive and tuned context."""
    selected_model = model or os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_0")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    return ChatOllama(
        model=selected_model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=REQUEST_TIMEOUT,
        keep_alive="30m",
        num_ctx=num_ctx,
    )


def get_structured_llm(schema):
    """
    Return an LLM that forces output matching the given Pydantic schema.

    Falls back to JSON-mode prompting if the Ollama version does not
    support native tool-calling structured output.
    """
    llm = get_llm()
    try:
        return llm.with_structured_output(schema)
    except Exception:
        # Older Ollama builds: return plain LLM; callers handle JSON parsing
        return llm
