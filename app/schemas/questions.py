"""Public API request and response schemas.

These Pydantic models define the external HTTP contract. They intentionally
mirror the service-layer DTOs they wrap without inheriting from them, so the
API contract can stay stable while internal domain models evolve.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints


class QuestionRequest(BaseModel):
    """Request body for submitting a question.

    Attributes:
        question: The question to process.
    """

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    ]


class Citation(BaseModel):
    """A source excerpt supporting a generated answer."""

    source_id: str
    chunk_id: str
    text: str


class RetrievalMetrics(BaseModel):
    """Counts describing retrieval work for a question."""

    documents_considered: int
    chunks_returned: int


class ModelMetrics(BaseModel):
    """Provider, model, and token usage for answer generation."""

    provider: str
    name: str
    input_tokens: int
    output_tokens: int


class QuestionResult(BaseModel):
    """Completed answer and supporting metadata."""

    request_id: str
    answer: str
    citations: list[Citation]
    retrieval: RetrievalMetrics
    model: ModelMetrics


class QuestionJobResponse(BaseModel):
    """Public response shape for polling a submitted question.

    Attributes:
        request_id: Unique identifier of the question job.
        status: Whether processing is pending or completed.
        result: Completed answer and metadata, or ``None`` while pending.
    """

    request_id: str
    status: Literal["pending", "completed", "failed"]
    result: QuestionResult | None = None
    # Controlled category only; provider exception details stay in server logs.
    error_category: Literal["processing_error"] | None = None
