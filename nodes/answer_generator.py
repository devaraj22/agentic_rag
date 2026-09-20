"""
nodes/answer_generator.py
--------------------------
LangGraph node: generate_answer

Synthesises PDF evidence, web evidence, conversation history, and the
evidence grade into a grounded, sourced answer.

Rules enforced:
  1. No hallucination — every claim must be traceable to retrieved evidence.
  2. Sources are distinguished ("According to your uploaded document…" vs
     "According to recent web sources…").
  3. For HYBRID queries, information is synthesised — not concatenated.
  4. Contradictions between sources are surfaced explicitly.
  5. Insufficient evidence is acknowledged rather than hidden.
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from llm_provider import get_llm

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 8000


GENERATOR_SYSTEM = """You are a precise, citation-aware research assistant.

Your response MUST follow these rules without exception:

RULE 1 — GROUNDING
Only use information that is explicitly present in the provided context.
Never invent, extrapolate, or assume facts not in the context.

RULE 2 — SOURCE ATTRIBUTION
Always distinguish your sources:
- For content from the uploaded PDF, say: "According to your uploaded document…"
  or "The document states…" or "[PDF: filename, p.X]"
- For content from web sources, say: "According to recent web sources…"
  or "A web source reports…" or "[Web: title]"
Never blend or paraphrase sources without attribution.

RULE 3 — HYBRID SYNTHESIS
For questions that use both PDF and web sources:
- Synthesise the information into a coherent answer.
- Explicitly connect what the document says to what current sources say.
- Do not simply concatenate a PDF section and a web section.

RULE 4 — CONTRADICTIONS
If the PDF and web sources disagree, surface it explicitly:
"The uploaded document states X, while a more recent web source reports Y."
Never silently pick one side.

RULE 5 — INSUFFICIENT EVIDENCE
ONLY if the evidence score is below 0.40 or evidence is completely missing:
Begin your response with: "⚠️ I could not find sufficient evidence to answer this confidently."
Otherwise, if evidence is sufficient (score >= 0.40), DO NOT include any warning about insufficient evidence.

RULE 6 — CONVERSATION CONTEXT
Use prior chat history to understand follow-up questions and avoid repeating
information already given.

FORMAT
- Write in clear, well-structured prose.
- Use bullet points or numbered lists only when the content is inherently list-like.
- Keep the answer focused and appropriately detailed for the question.
"""

GENERATOR_HUMAN = """
CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

ROUTING: {route} (reason: {route_reason})

EVIDENCE QUALITY: score={evidence_score:.2f}, sufficient={evidence_sufficient}
{grade_reason}

PDF EVIDENCE:
{pdf_evidence}

WEB EVIDENCE:
{web_evidence}

Please answer the question following the rules above.
"""


def _format_chat_history(history: list) -> str:
    if not history:
        return "No prior conversation."
    lines = []
    for msg in history[-6:]:  # last 3 turns
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_docs(docs: list, source_type: str) -> str:
    if not docs:
        return "None retrieved."
    lines = []
    for i, doc in enumerate(docs, 1):
        if source_type == "pdf":
            ref = f"[PDF: {doc.get('filename', '?')}, p.{doc.get('page_number', '?')}]"
        else:
            ref = f"[Web: {doc.get('title', 'Web')} — {doc.get('url', '')}]"
        content = doc.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "…"
        lines.append(f"Source {i} {ref}:\n{content}")
    text = "\n\n".join(lines)
    if len(text) > MAX_CONTEXT_CHARS // 2:
        text = text[:MAX_CONTEXT_CHARS // 2] + "\n…[truncated]"
    return text


def generate_answer(state: dict) -> dict:
    """
    LangGraph node.

    Reads  : state["question"], state["chat_history"], state["route"],
             state["route_reason"], state["pdf_documents"],
             state["web_documents"], state["evidence_score"],
             state["evidence_sufficient"], state["grade_reason"]
    Writes : state["answer"]
    """
    question = state.get("question", "")
    chat_history = state.get("chat_history") or []
    route = state.get("route", "PDF_ONLY")
    route_reason = state.get("route_reason", "")
    pdf_docs = state.get("pdf_documents") or []
    web_docs = state.get("web_documents") or []
    evidence_score = state.get("evidence_score", 0.0)
    evidence_sufficient = state.get("evidence_sufficient", False)
    grade_reason = state.get("grade_reason", "")

    # Fast path if user is asking about a document but none is uploaded
    if route == "PDF_ONLY" and not state.get("has_document", False) and not pdf_docs:
        return {
            "answer": (
                "⚠️ **No PDF document is indexed yet.**\n\n"
                "Please upload a PDF using the sidebar and click **⚡ Process & Index PDF** "
                "to ask questions about your documents, or ask a general/current question to search the web."
            )
        }

    human_prompt = GENERATOR_HUMAN.format(
        chat_history=_format_chat_history(chat_history),
        question=question,
        route=route,
        route_reason=route_reason,
        evidence_score=evidence_score,
        evidence_sufficient=evidence_sufficient,
        grade_reason=grade_reason,
        pdf_evidence=_format_docs(pdf_docs, "pdf"),
        web_evidence=_format_docs(web_docs, "web"),
    )

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM),
            HumanMessage(content=human_prompt),
        ]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        if evidence_sufficient and evidence_score >= 0.40:
            answer = answer.replace("\n⚠️ I could not find sufficient evidence to answer this confidently.", "").strip()
            answer = answer.replace("⚠️ I could not find sufficient evidence to answer this confidently.", "").strip()
        logger.info("generate_answer: generated %d characters", len(answer))
        return {"answer": answer}

    except Exception as err:
        logger.error("generate_answer error: %s", err, exc_info=True)
        return {
            "answer": (
                "⚠️ I encountered an error while generating the answer. "
                f"Please check that Ollama is running. Error: {err}"
            )
        }
