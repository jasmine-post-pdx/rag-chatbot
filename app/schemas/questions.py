"""Public API request and response schemas.

These Pydantic models define the external HTTP contract. They intentionally
mirror the service-layer DTOs they wrap without inheriting from them, so the
API contract can stay stable while internal domain models evolve.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class QuestionCreate(BaseModel):
    """Request body for submitting a question."""

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
        Field(description="The question to process."),
    ]


class CitationResponse(BaseModel):
    """A source excerpt supporting a generated answer."""

    source_id: str = Field(description="Identifier of the source document.")
    chunk_id: str = Field(
        description="Identifier of the retrieved chunk within that source."
    )
    text: str = Field(description="Excerpt text used as supporting evidence.")


class RetrievalMetricsResponse(BaseModel):
    """Counts describing retrieval work for a question."""

    documents_considered: int = Field(
        description="Number of documents considered during retrieval."
    )
    chunks_returned: int = Field(
        description="Number of chunks returned to the generator."
    )


class ModelMetricsResponse(BaseModel):
    """Provider, model, and token usage for answer generation.

    Token counts are null when the provider did not report usage.
    """

    provider: str = Field(description="Generation provider identifier.")
    name: str = Field(description="Model identifier reported for the generation.")
    input_tokens: int | None = Field(
        description="Input token count, or null when omitted."
    )
    output_tokens: int | None = Field(
        description="Output token count, or null when omitted."
    )


class QuestionResultResponse(BaseModel):
    """Completed answer and supporting metadata."""

    question_id: str = Field(description="Unique identifier of the question job.")
    answer: str = Field(description="Generated answer text.")
    citations: list[CitationResponse] = Field(
        description="Supporting source excerpts."
    )
    retrieval: RetrievalMetricsResponse = Field(
        description="Counts describing retrieval work."
    )
    model: ModelMetricsResponse = Field(
        description="Provider, model, and token usage metadata."
    )


class QuestionJobResponse(BaseModel):
    """Public response shape for polling a submitted question."""

    question_id: str = Field(description="Unique identifier of the question job.")
    status: Literal["pending", "completed", "failed"] = Field(
        description="Whether processing is pending, completed, or failed."
    )
    result: QuestionResultResponse | None = Field(
        default=None,
        description="Completed answer and metadata, or null while pending or failed.",
    )
    # Controlled category only; provider exception details stay in server logs.
    error_category: Literal["processing_error"] | None = Field(
        default=None,
        description="Safe failure category when status is failed.",
    )
