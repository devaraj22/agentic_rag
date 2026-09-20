# 🔬 Agentic Hybrid RAG Research Assistant

> **Upgrade of:** PDF-RAG Intelligence chatbot  
> **Upgraded to:** LangGraph-powered Agentic Hybrid RAG system

---

## What Was Added

| Component | Original | Upgraded |
|-----------|----------|----------|
| LLM | Ollama (llama3:8b) | ✅ Same (unchanged) |
| Embeddings | nomic-embed-text via Ollama | ✅ Same (unchanged) |
| Vector DB | ChromaDB (persistent) | ✅ Same (unchanged) |
| PDF pipeline | LlamaIndex | ✅ Same (unchanged) |
| UI | Streamlit | ✅ Same (enhanced) |
| Orchestration | None (linear) | ✅ **LangGraph agentic graph** |
| Web search | None | ✅ **Tavily Search API** |
| Query routing | None | ✅ **PDF_ONLY / WEB_ONLY / HYBRID** |
| Evidence grading | None | ✅ **Scored 0.0–1.0** |
| Self-correction | None | ✅ **Retry loop (max 2)** |
| Citations | None | ✅ **Structured PDF + Web** |
| Conversation memory | Basic history | ✅ **Full multi-turn context** |

---

## Architecture

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           ↓
                  ┌──────────────────┐
                  │ classify_query   │  ← LLM structured output
                  └────────┬─────────┘
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
           PDF_ONLY     WEB_ONLY      HYBRID
              │            │            │
              ↓            ↓            ↓
         retrieve_pdf  search_web  retrieve_pdf
              │          (Tavily)   + search_web
              └────────────┼──────────────┘
                           ↓
                  ┌──────────────────┐
                  │  merge_evidence  │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │  grade_evidence  │  ← LLM structured output (0.0–1.0)
                  └────────┬─────────┘
                           ↓
                    Evidence enough?
                       /       \
                     YES        NO (< 2 retries)
                      │          │
                      ↓          ↓
                   generate   retry_search  ← query reformulation
                    answer      │
                      │    merge + grade again
                      └────┬─────┘
                           ↓
                  ┌──────────────────┐
                  │ build_citations  │
                  └────────┬─────────┘
                           ↓
                         END
```

---

## File Structure

```
agentic-rag/
│
├── app.py                    # Streamlit UI (enhanced, original pipeline intact)
├── agent_graph.py            # LangGraph graph definition & compilation
├── agent_state.py            # Shared AgentState TypedDict
├── schemas.py                # Pydantic models for structured LLM output
├── llm_provider.py           # Ollama ChatLLM factory (cached)
│
├── nodes/
│   ├── __init__.py
│   ├── query_classifier.py   # classify_query node
│   ├── pdf_retriever.py      # retrieve_pdf node (wraps LlamaIndex)
│   ├── web_searcher.py       # search_web node (Tavily)
│   ├── evidence_merger.py    # merge_evidence node
│   ├── evidence_grader.py    # grade_evidence node
│   ├── answer_generator.py   # generate_answer node
│   ├── retry_searcher.py     # retry_search node (query reformulation)
│   └── citation_builder.py   # build_citations node
│
├── data/                     # PDF storage (gitignored)
├── chroma_db/                # ChromaDB persistent storage (gitignored)
├── .env.example              # Environment variable template
├── .env                      # Your keys (gitignored — NEVER commit)
├── requirements.txt          # All dependencies
└── README.md                 # This file
```

---

## Setup

### 1. Prerequisites

- **Ollama** running locally with `llama3:8b-instruct-q4_0` pulled:
  ```bash
  ollama pull llama3:8b-instruct-q4_0
  ollama pull nomic-embed-text
  ```

- **Python 3.10+**

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
TAVILY_API_KEY=tvly-your-key-here   # from https://app.tavily.com/
```

> Without `TAVILY_API_KEY`, `WEB_ONLY` and `HYBRID` queries will surface a
> configuration error. `PDF_ONLY` queries work without it.

### 4. Run

```bash
streamlit run app.py
```

---

## Usage Guide

### PDF-only questions
> "What does the uploaded document say about transformers?"  
> "Summarise chapter 3."  
> "What dataset was used according to the PDF?"

The system routes these directly to the ChromaDB retriever.

### Web-only questions
> "What are the latest LangGraph releases?"  
> "What is the current OpenAI API pricing?"  
> "What happened in AI research this week?"

The system bypasses the PDF and queries Tavily directly.

### Hybrid questions
> "Compare the method in my PDF with current research."  
> "Is the technique in this paper still commonly used?"  
> "How has this topic evolved since this document was published?"

The system queries both sources and synthesises the response.

---

## Evidence Scoring

| Score | Meaning | Action |
|-------|---------|--------|
| 0.90–1.00 | Excellent | Generate answer |
| 0.70–0.89 | Acceptable | Generate answer |
| 0.40–0.69 | Weak | Retry search (up to 2×) |
| 0.00–0.39 | Insufficient | Retry search, then generate with caveat |

---

## Citation Format

**PDF sources:**
```
[PDF: research_paper.pdf, p.12]
```

**Web sources:**
```
[Web: Article Title](https://source.com/article)
```

---

## Key Design Decisions

1. **Zero replacement of working components.** The LlamaIndex/ChromaDB/Ollama
   pipeline is entirely preserved. LangGraph wraps it without touching it.

2. **Structured outputs over string parsing.** All LLM decisions (routing,
   grading, query reformulation) use Pydantic models via
   `.with_structured_output()`, making them deterministic and testable.

3. **Graceful degradation.** Every node has a heuristic fallback in case the
   LLM call fails (Ollama not running, timeout, etc.).

4. **Source provenance is never lost.** Every document carries `source_type`,
   filename, page number, and URL throughout the entire pipeline.

5. **No infinite loops.** `MAX_RETRIES = 2` is enforced in the conditional
   edge; the graph always terminates.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAVILY_API_KEY` | Yes (for web) | — | Tavily Search API key |
| `OLLAMA_MODEL` | No | `llama3:8b-instruct-q4_0` | Ollama model name |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TIMEOUT` | No | `360` | Request timeout in seconds |
