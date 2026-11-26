"""Citation validation against evidence selected for the model prompt.

Citation IDs are checked for membership in the selected evidence set only.
Passing validation does not prove that a cited chunk semantically supports a
claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.context import RankedChunk
from app.domain.errors import StructuredOutputError
from app.domain.model_outputs import AnswerV1
from app.domain.questions import Citation


def validate_and_map_citations(
    answer: AnswerV1,
    selected_chunks: Sequence[RankedChunk],
) -> list[Citation]:
    """Validate claim citation IDs against selected evidence and map to Citation rows.

    :param answer: Structured model answer with claims and citation chunk IDs.
    :param selected_chunks: Chunks actually supplied to the model in the prompt.
    :returns: Deduplicated public citations drawn from ``selected_chunks``.
    :raises StructuredOutputError: When citation IDs are invalid for the selected set."""
    if answer.insufficient_evidence:
        return []

    selected_by_id = {chunk.chunk_id: chunk for chunk in selected_chunks}
    if answer.claims and not selected_by_id:
        raise StructuredOutputError(
            "validation",
            "Structured model output failed: validation",
        )

    citations: list[Citation] = []
    seen: set[str] = set()
    for claim in answer.claims:
        for chunk_id in claim.citation_chunk_ids:
            chunk = selected_by_id.get(chunk_id)
            if chunk is None:
                raise StructuredOutputError(
                    "validation",
                    "Structured model output failed: validation",
                )
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            citations.append(
                Citation(
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                )
            )
    return citations
