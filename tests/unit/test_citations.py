"""Unit tests for Step 8 citation validation against selected evidence.

These tests target ``app.services.citations.validate_and_map_citations``.
They encode deterministic ID membership checks only; passing them does not
prove that a cited chunk semantically supports a claim.
"""

from __future__ import annotations

import pytest

from app.domain.context import RankedChunk
from app.domain.errors import StructuredOutputError
from app.domain.model_outputs import AnswerClaim, AnswerV1
from app.domain.questions import Citation
from app.services.citations import validate_and_map_citations


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "fastapi-guide",
) -> RankedChunk:
    """Build a ranked chunk for citation validation tests.

    :param chunk_id: Chunk identifier preserved on mapping.
    :param text: Excerpt text mapped onto public citations.
    :param source_id: Source document identifier.
    :returns: A ``RankedChunk`` instance."""
    return RankedChunk(source_id=source_id, chunk_id=chunk_id, text=text)


def _supported_answer(
    text: str,
    *claims: tuple[str, list[str]],
) -> AnswerV1:
    """Build a non-abstaining AnswerV1 with claim/citation pairs.

    :param text: Answer text.
    :param claims: Pairs of ``(claim_text, citation_chunk_ids)``.
    :returns: A validated ``AnswerV1``."""
    return AnswerV1(
        answer=text,
        claims=[
            AnswerClaim(claim=claim_text, citation_chunk_ids=chunk_ids)
            for claim_text, chunk_ids in claims
        ],
        insufficient_evidence=False,
    )


def test_validate_and_map_citations_accepts_selected_ids():
    """Map claims that cite selected chunk IDs onto public Citation rows."""
    selected = [
        _chunk("c1", "FastAPI uses Depends for injection."),
        _chunk("c2", "Lifespan closes shared clients.", source_id="fastapi-lifespan"),
    ]
    answer = _supported_answer(
        "FastAPI uses Depends.",
        ("Depends wires dependencies", ["c1"]),
    )

    citations = validate_and_map_citations(answer, selected)

    assert citations == [
        Citation(
            source_id="fastapi-guide",
            chunk_id="c1",
            text="FastAPI uses Depends for injection.",
        )
    ]


def test_validate_and_map_citations_rejects_invented_ids():
    """Reject citation chunk IDs that are not in the selected evidence set."""
    selected = [_chunk("c1", "FastAPI uses Depends for injection.")]
    answer = _supported_answer(
        "FastAPI uses Depends.",
        ("Depends wires dependencies", ["invented-chunk"]),
    )

    with pytest.raises(StructuredOutputError) as raised:
        validate_and_map_citations(answer, selected)

    assert raised.value.category == "validation"


def test_validate_and_map_citations_rejects_retrieved_but_not_selected_ids():
    """Reject IDs present in a broader retrieved set but not in selected_chunks.

    Step 8 requires validation against chunks actually supplied to the model,
    not merely all retrieved candidates.
    """
    retrieved = [
        _chunk("c1", "Selected evidence about Depends."),
        _chunk("c2", "Retrieved but discarded by the budget."),
    ]
    selected = [retrieved[0]]
    answer = _supported_answer(
        "Answer citing the discarded chunk.",
        ("Claim about discarded evidence", ["c2"]),
    )

    with pytest.raises(StructuredOutputError) as raised:
        validate_and_map_citations(answer, selected)

    assert raised.value.category == "validation"
    # Sanity: the ID exists in the broader retrieved set.
    assert any(chunk.chunk_id == "c2" for chunk in retrieved)


def test_validate_and_map_citations_allows_abstention_with_empty_list():
    """Abstention with empty claims returns no citations and does not raise."""
    selected = [_chunk("c1", "Some evidence that was not enough.")]
    answer = AnswerV1(
        answer="I do not have enough evidence to answer.",
        claims=[],
        insufficient_evidence=True,
    )

    citations = validate_and_map_citations(answer, selected)

    assert citations == []


def test_validate_and_map_citations_rejects_claims_when_no_selected_evidence():
    """Fail closed when the model asserts claims but no evidence was selected."""
    answer = _supported_answer(
        "Unsupported claim without evidence.",
        ("A material claim", ["c1"]),
    )

    with pytest.raises(StructuredOutputError) as raised:
        validate_and_map_citations(answer, [])

    assert raised.value.category == "validation"


def test_validate_and_map_citations_deduplicates_shared_chunk_ids():
    """Emit one Citation row when multiple claims cite the same selected chunk."""
    selected = [_chunk("c1", "FastAPI uses Depends for injection.")]
    answer = _supported_answer(
        "FastAPI uses Depends for DI.",
        ("Depends injects dependencies", ["c1"]),
        ("Path operations receive shared resources", ["c1"]),
    )

    citations = validate_and_map_citations(answer, selected)

    assert citations == [
        Citation(
            source_id="fastapi-guide",
            chunk_id="c1",
            text="FastAPI uses Depends for injection.",
        )
    ]


def test_validate_and_map_citations_is_deterministic_id_check_not_semantic():
    """ID membership alone is accepted even when the claim text does not match.

    This documents the Step 8 distinction: deterministic citation validation is
    not evidence of semantic groundedness. A claim can cite a selected chunk
    whose text does not support the claim; the validator must not require
    entailment.
    """
    selected = [_chunk("c1", "Express middleware runs in order for each request.")]
    answer = _supported_answer(
        "FastAPI uses quantum entanglement for routing.",
        ("Quantum routing is built into FastAPI", ["c1"]),
    )

    citations = validate_and_map_citations(answer, selected)

    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"
    assert "quantum" not in citations[0].text.casefold()
    assert "Express middleware" in citations[0].text


def test_validate_and_map_citations_maps_text_from_selected_chunk_only():
    """Citation text and source_id come from the selected chunk, not the claim."""
    selected = [
        _chunk(
            "rag-budget:0",
            "Context budgeting reserves tokens for ranked chunks.",
            source_id="rag-budget",
        )
    ]
    answer = _supported_answer(
        "Budgeting reserves tokens.",
        ("Budgeting reserves tokens for ranked chunks", ["rag-budget:0"]),
    )

    citations = validate_and_map_citations(answer, selected)

    assert citations[0].source_id == "rag-budget"
    assert citations[0].chunk_id == "rag-budget:0"
    assert citations[0].text == "Context budgeting reserves tokens for ranked chunks."
