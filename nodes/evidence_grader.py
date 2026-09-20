"""
nodes/evidence_grader.py
------------------------
LangGraph node: grade_evidence

Determines whether the retrieved evidence is sufficient to answer the question.
Uses structured LLM output; falls back to a length/keyword heuristic when
the LLM is unavailable.

Threshold: evidence_score >= 0.70 is considered sufficient.
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage

from llm_provider import get_structured_llm
from schemas import EvidenceGrade

logger = logging.getLogger(__name__)

SUFFICIENCY_THRESHOLD = 0.70
MAX_EVIDENCE_CHARS = 6000  # Truncate to avoid context overflow

GRADER_SYSTEM = """You are an evidence quality evaluator for a research assistant.

Your job is to assess whether the retrieved context is sufficient to answer the user's question.

Evaluate the evidence on these criteria:
1. Relevance       – Does the evidence directly address the question?
2. Completeness    – Does it cover all aspects of the question?
3. Source quality  – Is the information from credible/authoritative sources?
4. Recency         – Is the information current enough for the question?
5. Contradictions  – Are there conflicting claims that make the answer uncertain?

Score guidelines:
  0.90 – 1.00 : Excellent — directly answers the question with strong support
  0.70 – 0.89 : Acceptable — answers the question adequately
  0.40 – 0.69 : Weak — partial answers, gaps present
  0.00 – 0.39 : Insufficient — evidence does not support a reliable answer

Set sufficient=True only if score >= 0.70.
In missing_information, state SPECIFICALLY what is absent (empty string if sufficient).
Respond in valid JSON format matching the schema with "sufficient", "score", "reason", and "missing_information" keys.
"""

GRADER_HUMAN = """Question: {question}

Retrieved Evidence:
{evidence_text}

Evaluate the evidence quality."""


def _format_evidence(evidence: list) -> str:
    """Render evidence list as structured text for the grader."""
    lines = []
    for i, doc in enumerate(evidence, 1):
        src = doc.get("source_type", "unknown").upper()
        if src == "PDF":
            ref = f"[PDF: {doc.get('filename', '?')}, p.{doc.get('page_number', '?')}]"
        else:
            ref = f"[WEB: {doc.get('title', 'Web')} — {doc.get('url', '')}]"
        lines.append(f"--- Source {i} {ref} ---\n{doc.get('content', '')}")
    text = "\n\n".join(lines)
    # Truncate to avoid context overflow
    if len(text) > MAX_EVIDENCE_CHARS:
        text = text[:MAX_EVIDENCE_CHARS] + "\n...[truncated]"
    return text


def _heuristic_grade(question: str, evidence: list) -> EvidenceGrade:
    """Simple fallback grader based on evidence volume."""
    total_chars = sum(len(d.get("content", "")) for d in evidence)
    if total_chars > 1500:
        return EvidenceGrade(sufficient=True, score=0.72, reason="Heuristic: sufficient volume of evidence retrieved.", missing_information="")
    elif total_chars > 400:
        return EvidenceGrade(sufficient=False, score=0.45, reason="Heuristic: limited evidence retrieved.", missing_information="More specific information may be needed.")
    else:
        return EvidenceGrade(sufficient=False, score=0.15, reason="Heuristic: very little evidence retrieved.", missing_information="Evidence is too sparse to answer reliably.")


def grade_evidence(state: dict) -> dict:
    """
    LangGraph node.

    Reads  : state["question"], state["evidence"]
    Writes : state["evidence_score"], state["evidence_sufficient"],
             state["grade_reason"], state["missing_information"]
    """
    question = state.get("question", "")
    evidence = state.get("evidence") or []

    if not evidence:
        logger.warning("grade_evidence: no evidence to grade")
        return {
            "evidence_score": 0.0,
            "evidence_sufficient": False,
            "grade_reason": "No evidence was retrieved.",
            "missing_information": "All relevant information is missing.",
        }

    fast_mode = state.get("fast_mode", False)
    if fast_mode:
        grade = _heuristic_grade(question, evidence)
        grade.sufficient = grade.score >= SUFFICIENCY_THRESHOLD
        logger.info(
            "grade_evidence (fast-mode): score=%.2f sufficient=%s | %s",
            grade.score,
            grade.sufficient,
            grade.reason,
        )
        return {
            "evidence_score": grade.score,
            "evidence_sufficient": grade.sufficient,
            "grade_reason": grade.reason,
            "missing_information": grade.missing_information,
        }

    evidence_text = _format_evidence(evidence)

    try:
        structured_llm = get_structured_llm(EvidenceGrade)
        messages = [
            SystemMessage(content=GRADER_SYSTEM),
            HumanMessage(
                content=GRADER_HUMAN.format(
                    question=question, evidence_text=evidence_text
                )
            ),
        ]
        grade: EvidenceGrade = structured_llm.invoke(messages)

        # Enforce threshold independently of the LLM's boolean
        grade.sufficient = grade.score >= SUFFICIENCY_THRESHOLD

        logger.info(
            "grade_evidence: score=%.2f sufficient=%s | %s",
            grade.score,
            grade.sufficient,
            grade.reason,
        )
        return {
            "evidence_score": grade.score,
            "evidence_sufficient": grade.sufficient,
            "grade_reason": grade.reason,
            "missing_information": grade.missing_information,
        }

    except Exception as err:
        logger.warning("grade_evidence LLM failed (%s); using heuristic.", err)
        grade = _heuristic_grade(question, evidence)
        grade.sufficient = grade.score >= SUFFICIENCY_THRESHOLD
        return {
            "evidence_score": grade.score,
            "evidence_sufficient": grade.sufficient,
            "grade_reason": grade.reason,
            "missing_information": grade.missing_information,
        }
