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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def is_groq_active() -> bool:
    """Check whether Groq provider should be used."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    provider = os.getenv("LLM_PROVIDER", "groq" if groq_key else "ollama").lower()
    return provider == "groq" and bool(groq_key and not groq_key.startswith("gsk_your"))


# ── Factory ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0, model: str | None = None):
    """Return an active Chat model (Groq Cloud or Ollama Local)."""
    if is_groq_active():
        from langchain_groq import ChatGroq
        selected_model = model or os.getenv("GROQ_MODEL", GROQ_MODEL)
        return ChatGroq(
            model_name=selected_model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_retries=2,
        )

    # Ollama Local (tuned for 4GB RTX 2050 GPU)
    selected_model = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
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
    Uses json_mode on Groq or native tool-calling / JSON mode on Ollama.
    """
    if is_groq_active():
        from langchain_groq import ChatGroq
        selected_model = os.getenv("GROQ_MODEL", GROQ_MODEL)
        llm = ChatGroq(
            model_name=selected_model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            max_retries=2,
        )
        try:
            return llm.with_structured_output(schema, method="json_mode")
        except Exception:
            return llm.with_structured_output(schema)

    llm = get_llm(temperature=0.0)
    try:
        return llm.with_structured_output(schema)
    except Exception:
        return llm
