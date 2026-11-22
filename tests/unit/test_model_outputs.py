"""Unit tests for versioned classification and answer model schemas."""

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.model_outputs import AnswerClaim, AnswerV1, ClassificationV1


def test_classification_v1_accepts_valid_payload():
    """Accept a complete classification payload with nullable filters."""
    result = ClassificationV1(
        intent="answer",
        needs_retrieval=True,
        framework="fastapi",
        language="python",
        confidence=0.8,
        clarifying_question=None,
    )

    assert result.schema_version == "1"
    assert result.intent == "answer"
    assert result.needs_retrieval is True
    assert result.framework == "fastapi"
    assert result.language == "python"
    assert result.confidence == 0.8
    assert result.clarifying_question is None


def test_classification_v1_rejects_unknown_intent_enum():
    """Reject an intent value outside the documented enum."""
    with pytest.raises(ValidationError):
        ClassificationV1(
            intent="search_admin",
            needs_retrieval=False,
            confidence=0.5,
        )


def test_classification_v1_rejects_extra_field():
    """Reject classification payloads that include undeclared fields."""
    with pytest.raises(ValidationError):
        ClassificationV1.model_validate(
            {
                "intent": "clarify",
                "needs_retrieval": False,
                "confidence": 0.4,
                "clarifying_question": "Which framework?",
                "admin": True,
            }
        )


def test_classification_v1_rejects_confidence_outside_unit_interval():
    """Reject confidence values outside the model-claim unit interval."""
    with pytest.raises(ValidationError):
        ClassificationV1(
            intent="out_of_scope",
            needs_retrieval=False,
            confidence=1.5,
        )


def test_answer_v1_accepts_supported_claims_with_citations():
    """Accept an answer that cites chunk IDs for each material claim."""
    result = AnswerV1(
        answer="FastAPI uses Pydantic for validation.",
        claims=[
            AnswerClaim(
                claim="FastAPI uses Pydantic",
                citation_chunk_ids=["chunk-1"],
            )
        ],
        insufficient_evidence=False,
    )

    assert result.schema_version == "1"
    assert result.insufficient_evidence is False
    assert result.claims[0].citation_chunk_ids == ["chunk-1"]


def test_answer_v1_accepts_insufficient_evidence_with_empty_claims():
    """Allow abstention when no claims are asserted."""
    result = AnswerV1(
        answer="I do not have enough evidence to answer.",
        claims=[],
        insufficient_evidence=True,
    )

    assert result.insufficient_evidence is True
    assert result.claims == []


def test_answer_v1_rejects_extra_field():
    """Reject answer payloads that include undeclared fields."""
    with pytest.raises(ValidationError):
        AnswerV1.model_validate(
            {
                "answer": "Hello",
                "claims": [],
                "insufficient_evidence": True,
                "secret": "nope",
            }
        )


def test_answer_v1_rejects_missing_citations_for_supported_claim():
    """Reject a supported claim that lists no citation chunk IDs."""
    with pytest.raises(ValidationError, match="citation_chunk_id"):
        AnswerV1(
            answer="Unsupported claim text",
            claims=[AnswerClaim(claim="A material claim", citation_chunk_ids=[])],
            insufficient_evidence=False,
        )


def test_answer_v1_rejects_claims_when_insufficient_evidence():
    """Reject claims that accompany an insufficient-evidence abstention."""
    with pytest.raises(ValidationError, match="claims must be empty"):
        AnswerV1(
            answer="Abstaining",
            claims=[
                AnswerClaim(claim="Should not appear", citation_chunk_ids=["c1"])
            ],
            insufficient_evidence=True,
        )


def test_schema_version_change_rejects_prior_version_payload():
    """A bumped schema version rejects payloads that still declare version 1."""

    class AnswerV2(BaseModel):
        """Local stand-in for a deliberately versioned successor schema."""

        model_config = ConfigDict(extra="forbid")

        schema_version: Literal["2"]
        answer: str
        claims: list[AnswerClaim]
        insufficient_evidence: bool

    with pytest.raises(ValidationError):
        AnswerV2.model_validate(
            {
                "schema_version": "1",
                "answer": "Still on v1",
                "claims": [],
                "insufficient_evidence": True,
            }
        )
