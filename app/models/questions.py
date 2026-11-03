"""Domain models for question processing and job state."""

from typing import Literal

from pydantic import BaseModel


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


class QuestionJob(BaseModel):
    """Current state of a submitted question."""

    request_id: str
    status: Literal["pending", "completed", "failed"]
    result: QuestionResult | None = None
    error_category: Literal["processing_error"] | None = None
