"""Citation validation against evidence selected for the model prompt.

Step 8 stubs: the public function exists so TDD tests can collect and fail red
before the real validation logic is implemented.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.context import RankedChunk
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
    :raises StructuredOutputError: When citation IDs are invalid for the selected set.
    :raises NotImplementedError: Until Step 8 citation validation is implemented."""
    raise NotImplementedError("Step 8 citation validation is not implemented yet")
