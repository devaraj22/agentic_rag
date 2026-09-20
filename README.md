# 🔬 Agentic Hybrid RAG Research Assistant

A production-grade, multi-engine Agentic RAG research assistant built with **LangGraph**, **LlamaIndex**, **ChromaDB**, **Groq Cloud API**, **Ollama**, and **Streamlit**.

The assistant dynamically classifies user queries, routes between local PDF documents and live web search, evaluates evidence relevance with self-correcting retry loops, and synthesizes answers with transparent source citations.

---

## Key Features & Comparison

| Capability | Baseline PDF-RAG | Agentic Hybrid RAG |
|---|---|---|
| **Inference Engine** | Local Ollama only | **Dual-Engine**: Groq Cloud API (~1 s) & Local Ollama |
| **Embeddings** | `nomic-embed-text` via Ollama | `nomic-embed-text` via Ollama |
| **Vector Database** | ChromaDB (persistent) | ChromaDB (persistent) |
| **Document Indexing** | LlamaIndex pipeline | LlamaIndex pipeline (fully preserved) |
| **Agent Orchestration** | Linear execution | **LangGraph State Graph** with cyclic retry logic |
| **Information Retrieval** | PDF documents only | **Hybrid**: PDF documents + Tavily live web search |
| **Query Routing** | Static / manual | **Dynamic routing**: `PDF_ONLY`, `WEB_ONLY`, `HYBRID` |
| **Evidence Grading** | None | **Automated scoring (0.0 – 1.0)** with quality gating |
| **Self-Correction** | None | **Autonomous query reformulation** (up to 2 retries) |
| **Citation Support** | None | **Transparent citations** with page numbers & URLs |
| **User Interface** | Basic Streamlit | **Professional Dark Theme** with route & evidence telemetry |

---

## Architecture & Workflow

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  classify_query  │  ◄── Fast-path heuristics or LLM router
                  └────────┬─────────┘
                           │
               ┌───────────┼───────────┐
               │           │           │
               ▼           ▼           ▼
           PDF_ONLY     WEB_ONLY     HYBRID
               │           │           │
               ▼           ▼           ▼
          retrieve_pdf search_web retrieve_pdf
               │        (Tavily)   + search_web
               └───────────┬───────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  merge_evidence  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  grade_evidence  │  ◄── Quality threshold check (0.0 – 1.0)
                  └────────┬─────────┘
                           │
                     Evidence sufficient?
                        /       \
                     YES         NO (< 2 retries)
                      │           │
                      ▼           ▼
                   generate  retry_search ◄── Autonomous query reformulation
                    answer        │
                      │      merge + grade again
                      └─────┬─────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ build_citations  │
                  └────────┬─────────┘
                           │
                           ▼
                         [END]
```

---

## Project Structure

```
agentic-rag/
├── app.py                   # Streamlit dark-mode interface & chat runner
├── agent_graph.py           # LangGraph graph topology, conditional edges & compilation
├── agent_state.py           # TypedDict state definition with channel reducers
├── schemas.py               # Pydantic schemas for structured LLM outputs
├── llm_provider.py          # Unified dual-engine factory (Groq Cloud & Local Ollama)
│
├── nodes/                   # Modular LangGraph execution nodes
│   ├── __init__.py
│   ├── query_classifier.py  # Route intent classifier (PDF_ONLY / WEB_ONLY / HYBRID)
│   ├── pdf_retriever.py     # ChromaDB / LlamaIndex vector retriever
│   ├── web_searcher.py      # Tavily / DuckDuckGo web searcher
│   ├── evidence_merger.py   # Unified evidence deduplicator and normalizer
│   ├── evidence_grader.py   # Relevance & sufficiency scoring node
│   ├── answer_generator.py  # Final grounded answer synthesis
│   ├── retry_searcher.py    # Query reformulation for low-scoring evidence
│   └── citation_builder.py  # Structured source citation generator
│
├── data/                    # PDF document storage
├── chroma_db/               # Persistent ChromaDB vector store
├── .streamlit/              # Streamlit configuration & theme tokens
│   └── config.toml
├── .env.example             # Environment configuration template
├── .env                     # Local API keys and runtime configuration
└── requirements.txt         # Project dependencies
```

---

## Quick Start & Setup

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed and serving:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  ```
- *(Recommended)* Free **Groq Cloud API Key** for ultra-fast ~1s inference: [console.groq.com](https://console.groq.com/)
- *(Optional)* Free **Tavily API Key** for live web search: [app.tavily.com](https://app.tavily.com/)

### 2. Installation

Clone repository and install dependencies:

```bash
git clone https://github.com/devaraj22/agentic_rag.git
cd agentic_rag
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment file:

```bash
cp .env.example .env
```

Populate the required credentials in `.env`:

```env
# Fast Cloud Inference (Recommended: ~1s response time)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Live Web Search
TAVILY_API_KEY=tvly-your_tavily_key_here

# Local Ollama Fallback
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## Inference Engine Options

| Mode | Engine | Average Latency | Hardware Utilization | Recommended Use |
|---|---|---|---|---|
| **Fast Cloud** | Groq Cloud API | **0.8 – 1.5 s** | Zero GPU load | Everyday research, high speed |
| **Local GPU** | Ollama (`llama3.2:3b`) | **3.0 – 5.0 s** | 100% GPU offload (RTX 2050+) | Offline, fully private data |
| **Local CPU/GPU** | Ollama (`llama3:8b`) | **15.0 – 35.0 s** | Heavy VRAM | Deep analysis when offline |

> Toggle between **Groq Cloud** and **Ollama Local** at any time directly in the sidebar settings.

---

## Query Routing & Usage Guide

### 1. Document-Only Queries (`PDF_ONLY`)
- *"What are the primary findings in chapter 2?"*
- *"Summarize the methodology section of the uploaded paper."*
- Bypasses web search entirely and retrieves context strictly from the indexed ChromaDB store.

### 2. Live Web Queries (`WEB_ONLY`)
- *"What are the recent updates in LangGraph 0.2?"*
- *"What is current pricing for Claude 3.5 Sonnet?"*
- Bypasses local documents and executes real-time search across the web via Tavily.

### 3. Hybrid Synthesis Queries (`HYBRID`)
- *"Compare the framework proposed in my document with current industry standards."*
- *"Are the benchmarks reported in this paper still state-of-the-art?"*
- Queries both local document chunks and web results, unifying evidence before synthesizing the answer.

---

## Evidence Evaluation & Self-Correction

Each retrieved chunk is evaluated for relevance and semantic sufficiency:

| Evidence Score | Quality Band | System Action |
|---|---|---|
| **0.70 – 1.00** | High Quality | Proceed directly to answer generation |
| **0.40 – 0.69** | Moderate Quality | Query reformulation triggered; retries search (max 2 attempts) |
| **0.00 – 0.39** | Insufficient | Reformulates query; if still low after 2 retries, answers with caution notice |

---

## Transparent Citations

All answers include verified citations with direct source links or document page numbers:

- **PDF Documents**:
  ```text
  [PDF: attention_is_all_you_need.pdf, Page 4]
  ```
- **Web Sources**:
  ```text
  [Web: LangChain Documentation](https://python.langchain.com/docs/)
  ```

---

## Environment Variables Reference

| Variable | Type | Required | Default | Purpose |
|---|---|---|---|---|
| `GROQ_API_KEY` | String | Optional | — | API key for high-speed Groq Cloud inference |
| `GROQ_MODEL` | String | Optional | `openai/gpt-oss-20b` | Model name on Groq platform |
| `TAVILY_API_KEY` | String | Optional | — | API key for Tavily live web search |
| `OLLAMA_MODEL` | String | Optional | `llama3.2:3b` | Local Ollama model identifier |
| `OLLAMA_BASE_URL` | String | Optional | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_TIMEOUT` | Float | Optional | `120` | Request timeout in seconds |
| `FAST_MODE` | Boolean | Optional | `true` | Enables fast-path heuristic routing and grading |

---

## License

MIT License. See [LICENSE](file:///d:/agentic-rag-research-assistant/agentic-rag/LICENSE) for details.
