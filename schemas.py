"""
schemas.py
----------
Pydantic models for structured LLM outputs.
Using structured output prevents fragile string-parsing of LLM responses.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class QueryClassification(BaseModel):
    """Output of the classify_query node."""

    route: Literal["PDF_ONLY", "WEB_ONLY", "HYBRID"] = Field(
        description=(
            "PDF_ONLY  – question is entirely answerable from the uploaded document. "
            "WEB_ONLY  – question requires live/current information not in any PDF. "
            "HYBRID    – question needs both the uploaded document AND live web context."
        )
    )
    reason: str = Field(
        description="One-sentence explanation for why this route was chosen."
    )


class EvidenceGrade(BaseModel):
    """Output of the grade_evidence node."""

    sufficient: bool = Field(
        description=(
            "True if the retrieved evidence is relevant and complete enough "
            "to answer the question reliably."
        )
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score: "
            "0.90–1.0 = strong evidence, "
            "0.70–0.89 = acceptable, "
            "0.40–0.69 = weak, "
            "0.00–0.39 = insufficient."
        ),
    )
    reason: str = Field(
        description="Explanation of what is good or lacking in the evidence."
    )
    missing_information: str = Field(
        description=(
            "What specific information is absent or unclear. "
            "Empty string if evidence is sufficient."
        )
    )


class RetryQuery(BaseModel):
    """Output of the retry_search reformulation step."""

    reformulated_query: str = Field(
        description=(
            "An improved, more specific search query that addresses the "
            "identified missing information."
        )
    )
    reason: str = Field(
        description="Why this reformulation should yield better evidence."
    )
