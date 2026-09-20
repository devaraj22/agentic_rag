"""
app.py — Agentic Hybrid RAG Research Assistant
------------------------------------------------
Streamlit UI — dark mode, professional redesign.
"""

import os
import sys
import logging
import time

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

# ── 1. SYSTEM CONFIGURATION ───────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_0")
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# ── 2. PROMPT TEMPLATE ────────────────────────────────────────────────────────
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


# ── 3. PDF RAG INITIALISER ────────────────────────────────────────────────────
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


# ── 5. PAGE CONFIG & DARK THEME ───────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Dark background ── */
.stApp { background-color: #0f1117 !important; color: #e2e8f0 !important; }

/* ── Top Bar & Deploy Button ── */
#MainMenu, footer, .stDeployButton, [data-testid="stDeployButton"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background-color: transparent !important;
    pointer-events: none;
}
header[data-testid="stHeader"] * {
    pointer-events: auto;
}

/* ── Main container ── */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 920px;
}

/* ── Dark sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b27 !important;
    border-right: 1px solid #2a2f3e !important;
    min-width: 260px !important;
    max-width: 280px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.2rem 1rem 1.5rem 1rem !important;
}

/* ── Sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }

/* ── Sidebar selectbox ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: #1e2535 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: #94a3b8 !important; }

/* ── Sidebar file uploader ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
    background-color: #1e2535 !important;
    border: 1.5px dashed #2a3a5c !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span { color: #94a3b8 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #252d40 !important;
    border: 1px solid #2a3a5c !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* ── Sidebar radio ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #cbd5e0 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px !important; }

/* ── Sidebar toggle ── */
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #cbd5e0 !important;
    font-size: 0.85rem !important;
}

/* ── Sidebar section labels ── */
.sidebar-section {
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; margin: 1.2rem 0 0.4rem 0;
}

/* ── Active file chip ── */
.file-chip {
    display: flex; align-items: center; gap: 6px;
    padding: 7px 10px; background: #0d2b20;
    border: 1px solid #065f46; border-radius: 8px;
    font-size: 0.8rem; color: #6ee7b7; font-weight: 500; margin-top: 6px;
}

/* ── Sidebar button ── */
.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    width: 100% !important;
    padding: 0.45rem 0.75rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.5rem;
}
.app-header-icon {
    font-size: 1.6rem;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    background: #1e2535;
    border: 1px solid #2e384d;
    border-radius: 12px;
    flex-shrink: 0;
}
.app-header-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.app-header-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    line-height: 1.25;
    letter-spacing: -0.01em;
}
.app-header-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 3px 0 0 0;
    line-height: 1.3;
}

/* ── Status pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.74rem; font-weight: 600;
}
.pill-green { background: #064e3b; color: #6ee7b7; }
.pill-blue  { background: #1e3a5f; color: #93c5fd; }
.pill-gray  { background: #1e2535; color: #94a3b8; }

/* ── Route badges ── */
.route-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 9px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 600; margin-bottom: 2px;
}
.route-pdf    { background: #1e3a5f; color: #93c5fd; border: 1px solid #2a4a7f; }
.route-web    { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
.route-hybrid { background: #2e1065; color: #c4b5fd; border: 1px solid #4c1d95; }

/* ── Evidence bar ── */
.ev-wrap  { margin: 5px 0 0 0; }
.ev-track { height: 4px; background: #1e2535; border-radius: 10px; overflow: hidden; }
.ev-fill  { height: 100%; border-radius: 10px; }
.ev-label { font-size: 0.72rem; color: #94a3b8; margin-top: 3px; }

/* ── Citation cards ── */
.cite-card {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 9px 13px; border-radius: 8px; margin-bottom: 6px;
    font-size: 0.82rem; line-height: 1.5;
}
.cite-card-pdf { background: #1a2540; border: 1px solid #2a3a5c; }
.cite-card-web { background: #0d2b20; border: 1px solid #065f46; }
.cite-icon  { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.cite-body  { flex: 1; color: #cbd5e0; }
.cite-score { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid #1a1f2e !important;
    padding: 0.5rem 0 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background-color: #1e2535 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }
[data-testid="stChatInput"] button { background: #2563eb !important; border-radius: 8px !important; }

/* ── Timing caption ── */
.time-cap { font-size: 0.72rem; color: #94a3b8; margin-top: 3px; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 4.5rem 2rem; }
.empty-state-icon  { font-size: 3rem; margin-bottom: 1rem; }
.empty-state-title { font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.5rem; }
.empty-state-sub   { font-size: 0.88rem; color: #94a3b8; line-height: 1.6; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b27 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #cbd5e1 !important; }

/* ── Status widget ── */
[data-testid="stStatus"] {
    background: #161b27 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
}

hr { border-color: #2a2f3e !important; margin: 0.75rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── 6. SESSION STATE ──────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None


# ── 7. SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#e2e8f0;'>⚙️ Settings</div>", unsafe_allow_html=True)

    # — Document —
    st.markdown("<div class='sidebar-section'>Document</div>", unsafe_allow_html=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)

    existing_pdfs = []
    for f in os.listdir("./data"):
        if f.lower().endswith(".pdf"):
            fp = os.path.join("./data", f)
            if os.path.isfile(fp) and os.path.getsize(fp) > 0 and f != "drone.pdf":
                existing_pdfs.append(f)

    if existing_pdfs:
        chosen = st.selectbox(
            "Existing PDFs",
            ["— Select —"] + existing_pdfs,
            index=0,
            label_visibility="collapsed",
        )
        if chosen != "— Select —" and st.button(f"Load  {chosen}"):
            file_path = os.path.join("./data", chosen)
            try:
                with st.spinner("Indexing…"):
                    st.session_state.query_engine = initialize_rag(file_path)
                    st.session_state.indexed_file = chosen
                st.success(f"Loaded: {chosen}")
            except Exception as exc:
                st.error(str(exc))

    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        file_path = os.path.join("./data", uploaded_file.name)
        with open(file_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        if st.button("Index Uploaded PDF"):
            try:
                with st.spinner("Embedding…"):
                    st.session_state.query_engine = initialize_rag(file_path)
                    st.session_state.indexed_file = uploaded_file.name
                st.success(f"Indexed: {uploaded_file.name}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.indexed_file:
        st.markdown(
            f"<div class='file-chip'>📎 {st.session_state.indexed_file}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No PDF loaded — web search will be used.")

    # — Engine —
    st.markdown("<div class='sidebar-section'>Inference Engine</div>", unsafe_allow_html=True)
    groq_available = bool(os.getenv("GROQ_API_KEY"))
    provider_options = (
        ["Groq Cloud  (~1.5 s)", "Ollama Local  (~4 s)"]
        if groq_available
        else ["Ollama Local  (~4 s)"]
    )
    selected_provider = st.radio(
        "Engine",
        provider_options,
        index=0 if os.getenv("LLM_PROVIDER", "groq") == "groq" else (len(provider_options) - 1),
    )
    os.environ["LLM_PROVIDER"] = "groq" if "Groq" in selected_provider else "ollama"

    fast_mode = st.toggle("⚡ Fast Mode", value=True,
                          help="Skip slow intermediate LLM calls when evidence confidence is high.")

    # — Status —
    st.markdown("<div class='sidebar-section'>Status</div>", unsafe_allow_html=True)
    groq_label = (
        "<span class='pill pill-blue'>⚡ Groq Cloud</span>"
        if bool(os.getenv("GROQ_API_KEY")) and os.getenv("LLM_PROVIDER", "groq") == "groq"
        else "<span class='pill pill-gray'>💻 Ollama 3B</span>"
    )
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    tavily_ok = bool(tavily_key and not tavily_key.startswith("tvly-your-key-here") and len(tavily_key) > 10)
    search_label = (
        "<span class='pill pill-green'>✅ Tavily</span>"
        if tavily_ok
        else "<span class='pill pill-gray'>🌐 DuckDuckGo</span>"
    )
    st.markdown(f"LLM &nbsp; {groq_label}", unsafe_allow_html=True)
    st.markdown(f"Search &nbsp; {search_label}", unsafe_allow_html=True)

    # — Session —
    st.markdown("<div class='sidebar-section'>Session</div>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(f"Model: `{OLLAMA_MODEL}` · Embeddings: `nomic-embed-text` · DB: ChromaDB")


# ── 8. HELPERS ────────────────────────────────────────────────────────────────
def _route_badge(route: str) -> str:
    cfg = {
        "PDF_ONLY": ("route-pdf",    "📄 PDF"),
        "WEB_ONLY": ("route-web",    "🌐 Web"),
        "HYBRID":   ("route-hybrid", "🔀 Hybrid"),
    }
    cls, label = cfg.get(route, ("route-pdf", route))
    return f"<span class='route-badge {cls}'>{label}</span>"


def _evidence_bar(score: float) -> str:
    pct   = int(score * 100)
    color = "#16a34a" if score >= 0.70 else ("#ca8a04" if score >= 0.40 else "#dc2626")
    return (
        f"<div class='ev-wrap'>"
        f"<div class='ev-track'><div class='ev-fill' style='width:{pct}%;background:{color}'></div></div>"
        f"<div class='ev-label'>Evidence quality: <b>{pct}%</b></div>"
        f"</div>"
    )


def _render_citations(citations: list):
    for cite in citations:
        if cite["type"] == "pdf":
            st.markdown(
                f"<div class='cite-card cite-card-pdf'>"
                f"<div class='cite-icon'>📄</div>"
                f"<div class='cite-body'>{cite['label']}"
                f"<div class='cite-score'>Relevance: {cite['relevance_score']:.0%}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            url   = cite.get("url", "")
            title = cite.get("title", "Web Source")
            link  = f"<a href='{url}' target='_blank' style='color:#93c5fd;text-decoration:none;'>{title}</a>" if url else title
            st.markdown(
                f"<div class='cite-card cite-card-web'>"
                f"<div class='cite-icon'>🌐</div>"
                f"<div class='cite-body'>{link}"
                f"<div class='cite-score'>Relevance: {cite['relevance_score']:.0%}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── 9. PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
  <div class='app-header-icon'>🔬</div>
  <div class='app-header-text'>
    <div class='app-header-title'>Research Assistant</div>
    <div class='app-header-sub'>LangGraph · LlamaIndex · ChromaDB · Groq / Ollama · Tavily</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── 10. CHAT HISTORY ──────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-state-icon'>🔬</div>
        <div class='empty-state-title'>Research Assistant ready</div>
        <div class='empty-state-sub'>
            Upload a PDF in the sidebar for document Q&amp;A,<br>
            or just ask anything to search the web.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta  = msg["metadata"]
            route = meta.get("route", "")
            score = meta.get("evidence_score", 0.0)
            if route:
                st.markdown(
                    f"{_route_badge(route)} &nbsp; {_evidence_bar(score)}",
                    unsafe_allow_html=True,
                )
            parts = []
            if meta.get("elapsed_s"):
                parts.append(f"⏱ {meta['elapsed_s']}s")
            if meta.get("retry_count", 0):
                parts.append(f"🔄 retried {meta['retry_count']}×")
            if parts:
                st.markdown(f"<div class='time-cap'>{' · '.join(parts)}</div>", unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"Sources  ({len(msg['citations'])})", expanded=False):
                _render_citations(msg["citations"])


# ── 11. CHAT INPUT ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your document or the web…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response  = None
        t_elapsed = 0.0

        with st.status("Running…", expanded=True) as status_box:
            try:
                t_start = time.time()
                status_box.write("Gathering evidence…")
                response = run_agentic_query(
                    question=prompt,
                    query_engine=st.session_state.query_engine,
                    chat_history=st.session_state.chat_history[:-1],
                    fast_mode=fast_mode,
                )
                t_elapsed = round(time.time() - t_start, 1)
                status_box.update(label=f"Done in {t_elapsed}s", state="complete", expanded=False)
            except Exception as exc:
                err_msg = f"⚠️ {exc}"
                status_box.update(label="Error", state="error", expanded=False)
                st.error(err_msg)
                logger.error("app.py error: %s", exc, exc_info=True)
                response = {"answer": err_msg, "citations": [], "metadata": {}}

        if response:
            answer    = response.get("answer", "No answer generated.")
            citations = response.get("citations", [])
            meta      = response.get("metadata", {})
            meta["elapsed_s"] = t_elapsed
            route     = meta.get("route", "")
            score     = meta.get("evidence_score", 0.0)
            retries   = meta.get("retry_count", 0)

            st.markdown(answer)

            if route:
                st.markdown(
                    f"{_route_badge(route)} &nbsp; {_evidence_bar(score)}",
                    unsafe_allow_html=True,
                )

            parts = []
            if t_elapsed:
                parts.append(f"⏱ {t_elapsed}s")
            if retries:
                parts.append(f"🔄 retried {retries}×")
            if parts:
                st.markdown(f"<div class='time-cap'>{' · '.join(parts)}</div>", unsafe_allow_html=True)

            if citations:
                with st.expander(f"Sources  ({len(citations)})", expanded=False):
                    _render_citations(citations)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "metadata": meta,
            })
