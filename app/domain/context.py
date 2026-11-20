"""Domain models for context budgeting before generation.

These types describe ranked retrieval candidates and the deterministic outcome
of allocating them into a model context window. They are not part of the
public HTTP contract.
"""

from pydantic import BaseModel, Field


class RankedChunk(BaseModel):
    """A retrieval candidate considered for inclusion in the prompt.

    :ivar source_id: Identifier of the source document.
    :ivar chunk_id: Identifier of the chunk within that source.
    :ivar text: Chunk text that would be supplied as evidence.
    :ivar rank: Optional rank position; lower values are preferred when present.
    :ivar score: Optional retrieval score; unused by greedy token selection."""

    source_id: str
    chunk_id: str
    text: str
    rank: int | None = None
    score: float | None = None


class BudgetRequest(BaseModel):
    """Fixed and retrieval inputs for a context-budget allocation.

    :ivar question: User question text.
    :ivar instructions: Optional system/instructions text counted as fixed input.
    :ivar history: Prior turn texts counted as fixed input, in order.
    :ivar tool_schemas: Serialized tool schema strings counted as fixed input.
    :ivar ranked_chunks: Retrieval candidates in preferred selection order."""

    question: str
    instructions: str | None = None
    history: list[str] = Field(default_factory=list)
    tool_schemas: list[str] = Field(default_factory=list)
    ranked_chunks: list[RankedChunk] = Field(default_factory=list)


class BudgetAllocation(BaseModel):
    """Deterministic result of allocating chunks into the context budget.

    :ivar selected_chunks: Chunks chosen in rank order without exceeding capacity.
    :ivar eligible_count: Number of ranked chunks considered.
    :ivar selected_count: Number of chunks included in the assembled prompt.
    :ivar discarded_count: Number of chunks skipped for not fitting the budget.
    :ivar available_retrieval_tokens: Token capacity reserved for framing and chunks.
    :ivar estimated_fixed_tokens: Tokens counted for non-retrieval fixed input
        plus output reserve and safety margin.
    :ivar estimated_prompt_tokens: Tokens counted for the assembled prompt text
        (and instructions when present), excluding output reserve and margin.
    :ivar is_estimate: ``True`` when the token counter labeled counts as estimates.
    :ivar assembled_prompt: Prompt text ready to send as generation input."""

    selected_chunks: list[RankedChunk]
    eligible_count: int
    selected_count: int
    discarded_count: int
    available_retrieval_tokens: int
    estimated_fixed_tokens: int
    estimated_prompt_tokens: int
    is_estimate: bool
    assembled_prompt: str
