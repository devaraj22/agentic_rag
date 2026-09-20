"""
app.py — Agentic Hybrid RAG Research Assistant
------------------------------------------------
Streamlit UI layer.

What changed from the original app.py
---------------------------------------
KEPT: LlamaIndex + ChromaDB + Ollama pipeline (initialize_rag, QA_PROMPT_TMPL)
KEPT: Streamlit-based chat interface
KEPT: Per-PDF indexed collections in ChromaDB
ADDED: LangGraph agentic brain (agent_graph.build_graph)
ADDED: Tavily web search with TAVILY_API_KEY from .env
ADDED: Route display badge (PDF_ONLY / WEB_ONLY / HYBRID)
ADDED: Evidence quality score bar
ADDED: Structured citations panel
ADDED: Multi-turn conversation memory (full history passed to graph)
ADDED: Retry count display
"""

import os
import sys
import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── 1. SYSTEM CONFIGURATION (original — unchanged) ───────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_0")
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# ── 2. PROMPT TEMPLATE (original — unchanged) ────────────────────────────────
SYSTEM_PROMPT = (
    "### ROLE\n"
    "You are a highly specialized Document Intelligence Assistant. Your primary function is to "
    "interpret data from provided PDF excerpts and answer user questions accurately.\n\n"
    "### OPERATIONAL CONSTRAINTS\n"
    "1. Context Adherence: Use ONLY the information provided in the <context> tags.\n"
    "2. Missing Information: If the answer is not in the context, say: "
    "'I'm sorry, I cannot find info in the document.'\n"
    "3. Language: If the user uses Tanglish, explain concepts in Tanglish mix for clarity.\n\n"
    "### CONTEXT\n"
    "{context_str}\n\n"
    "### USER QUERY\n"
    "{query_str}\n\n"
    "### RESPONSE\n"
)
QA_PROMPT_TMPL = PromptTemplate(SYSTEM_PROMPT)


# ── 3. PDF RAG INITIALISER (original — unchanged) ────────────────────────────
def initialize_rag(file_path: str):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    if not documents or not any(doc.text.strip() for doc in documents):
        raise ValueError(f"No readable text could be extracted from '{os.path.basename(file_path)}'.")
    db = chromadb.PersistentClient(path="./chroma_db")
    coll_name = "".join(filter(str.isalnum, os.path.basename(file_path)))[:60]
    chroma_collection = db.get_or_create_collection(coll_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    query_engine = index.as_query_engine(
        text_qa_template=QA_PROMPT_TMPL, similarity_top_k=3
    )
    return query_engine


# ── 4. AGENTIC GRAPH RUNNER ──────────────────────────────────────────────────
def run_agentic_query(question: str, query_engine, chat_history: list, fast_mode: bool = True) -> dict:
    from agent_graph import build_graph

    graph = build_graph(query_engine=query_engine)
    initial_state = {
        "question": question,
        "chat_history": chat_history,
        "has_document": query_engine is not None,
        "fast_mode": fast_mode,
        "retry_count": 0,
        "pdf_documents": [],
        "web_documents": [],
        "evidence": [],
    }
    result = graph.invoke(initial_state)
    return result.get(
        "final_response",
        {"answer": "No response generated.", "citations": [], "metadata": {}},
    )


# ── 5. PAGE CONFIG & STYLES ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    .badge-pdf    { background:#1d4ed8;color:#fff;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600; }
    .badge-web    { background:#065f46;color:#fff;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600; }
    .badge-hybrid { background:#7c3aed;color:#fff;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600; }
    .evidence-bar  { height:8px;border-radius:4px;background:#e5e7eb;margin:6px 0; }
    .evidence-fill { height:100%;border-radius:4px; }
    .cite-pdf { border-left:4px solid #1d4ed8;padding:6px 10px;margin:4px 0;background:#eff6ff;border-radius:0 6px 6px 0;font-size:.82rem; }
    .cite-web { border-left:4px solid #065f46;padding:6px 10px;margin:4px 0;background:#ecfdf5;border-radius:0 6px 6px 0;font-size:.82rem; }
    .stButton>button { width:100%;border-radius:8px;background:#2563eb;color:#fff;font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 6. HEADER ────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("🔬 Agentic RAG Research Assistant")
    st.caption("LangGraph · LlamaIndex · ChromaDB · Ollama · Tavily / DuckDuckGo")
with col_status:
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    tavily_ok = bool(tavily_key and not tavily_key.startswith("tvly-your-key-here") and len(tavily_key) > 10)
    if tavily_ok:
        st.metric("Web Search", "✅ Tavily")
    else:
        st.metric("Web Search", "🌐 DuckDuckGo (Free)")

# ── 7. SESSION STATE ──────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

# ── 8. SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("📄 Document")

    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)

    # List any valid existing PDFs in data/
    existing_pdfs = []
    for f in os.listdir("./data"):
        if f.lower().endswith(".pdf"):
            fp = os.path.join("./data", f)
            if os.path.isfile(fp) and os.path.getsize(fp) > 0 and f != "drone.pdf":
                existing_pdfs.append(f)

    if existing_pdfs:
        chosen = st.selectbox(
            "Select existing PDF:",
            ["(None / Upload new)"] + existing_pdfs,
            index=0,
        )
        if chosen != "(None / Upload new)" and st.button(f"⚡ Load '{chosen}'"):
            file_path = os.path.join("./data", chosen)
            try:
                with st.spinner(f"Indexing {chosen}…"):
                    st.session_state.query_engine = initialize_rag(file_path)
                    st.session_state.indexed_file = chosen
                st.success(f"✅ Loaded: {chosen}")
            except Exception as exc:
                st.error(f"Could not index '{chosen}': {exc}")

    uploaded_file = st.file_uploader("Or Upload PDF", type="pdf")

    if uploaded_file:
        file_path = os.path.join("./data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("⚡ Process & Index Uploaded PDF"):
            try:
                with st.spinner("Embedding document segments…"):
                    st.session_state.query_engine = initialize_rag(file_path)
                    st.session_state.indexed_file = uploaded_file.name
                st.success(f"✅ Indexed: {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Could not index uploaded file: {exc}")

    if st.session_state.indexed_file:
        st.info(f"📎 Active: **{st.session_state.indexed_file}**")
    else:
        st.warning("No PDF indexed yet (Web search will be used).")

    st.divider()
    st.subheader("⚡ Performance")
    fast_mode = st.toggle(
        "⚡ Fast Mode",
        value=True,
        help="Skips slow intermediate LLM evaluations when queries and evidence are high-confidence. Recommended for laptop GPUs (RTX 2050).",
    )

    st.divider()
    st.subheader("🧭 Routing Guide")
    st.markdown(
        "<span class='badge-pdf'>PDF ONLY</span> Questions about your document<br><br>"
        "<span class='badge-web'>WEB ONLY</span> Latest/current information<br><br>"
        "<span class='badge-hybrid'>HYBRID</span> Document + web comparison",
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(f"LLM: `{OLLAMA_MODEL}`")
    st.caption("Embeddings: `nomic-embed-text`")
    st.caption("Vector DB: ChromaDB (persistent)")


# ── 9. HELPERS ────────────────────────────────────────────────────────────────
def _route_badge(route: str) -> str:
    cls_map   = {"PDF_ONLY": "badge-pdf", "WEB_ONLY": "badge-web", "HYBRID": "badge-hybrid"}
    label_map = {"PDF_ONLY": "📄 PDF ONLY", "WEB_ONLY": "🌐 WEB ONLY", "HYBRID": "🔀 HYBRID"}
    return f"<span class='{cls_map.get(route,'badge-pdf')}'>{label_map.get(route,route)}</span>"


def _evidence_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#16a34a" if score >= 0.70 else ("#ca8a04" if score >= 0.40 else "#dc2626")
    return (
        f"<div class='evidence-bar'><div class='evidence-fill' "
        f"style='width:{pct}%;background:{color};'></div></div>"
        f"<small style='color:#6b7280'>Evidence quality: <b>{pct}%</b></small>"
    )


def _render_citations(citations: list):
    for cite in citations:
        if cite["type"] == "pdf":
            st.markdown(
                f"<div class='cite-pdf'>📄 {cite['label']}"
                f" &nbsp; <i>relevance: {cite['relevance_score']:.2f}</i></div>",
                unsafe_allow_html=True,
            )
        else:
            url = cite.get("url", "")
            title = cite.get("title", "Web Source")
            link = f"<a href='{url}' target='_blank'>{title}</a>" if url else title
            st.markdown(
                f"<div class='cite-web'>🌐 {link}"
                f" &nbsp; <i>relevance: {cite['relevance_score']:.2f}</i></div>",
                unsafe_allow_html=True,
            )


# ── 10. CHAT HISTORY DISPLAY ──────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            st.markdown(_route_badge(meta.get("route", "")), unsafe_allow_html=True)
            st.markdown(_evidence_bar(meta.get("evidence_score", 0.0)), unsafe_allow_html=True)
            if meta.get("elapsed_s"):
                st.caption(f"⏱️ Generated in {meta['elapsed_s']}s")
            if meta.get("retry_count", 0):
                st.caption(f"🔄 Search retried {meta['retry_count']}×")
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])})", expanded=False):
                _render_citations(msg["citations"])


# ── 11. CHAT INPUT ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything — about your document or the web…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🚀 Running agentic pipeline…", expanded=True) as status_box:
            try:
                import time
                t_start = time.time()
                status_box.write("🧭 Evaluating query & gathering evidence…")
                response = run_agentic_query(
                    question=prompt,
                    query_engine=st.session_state.query_engine,
                    chat_history=st.session_state.chat_history[:-1],
                    fast_mode=fast_mode,
                )
                t_elapsed = round(time.time() - t_start, 1)
                status_box.update(label=f"✅ Finished in {t_elapsed}s", state="complete", expanded=False)

                answer    = response.get("answer", "No answer generated.")
                citations = response.get("citations", [])
                meta      = response.get("metadata", {})
                meta["elapsed_s"] = t_elapsed
                route     = meta.get("route", "")
                score     = meta.get("evidence_score", 0.0)
                retries   = meta.get("retry_count", 0)

                st.markdown(answer)
                st.markdown(_route_badge(route), unsafe_allow_html=True)
                st.markdown(_evidence_bar(score), unsafe_allow_html=True)
                st.caption(f"⏱️ Generated in **{t_elapsed}s**")
                if retries:
                    st.caption(f"🔄 Search automatically retried {retries}×")

                if citations:
                    with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
                        _render_citations(citations)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "metadata": meta,
                })

            except Exception as exc:
                err_msg = f"⚠️ An error occurred: {exc}"
                st.error(err_msg)
                logger.error("app.py error: %s", exc, exc_info=True)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
