"""Internal models for question processing and job state.

These DTOs are stored and passed through the workflow. Public HTTP models live
in ``app.schemas`` and do not inherit from these classes.
"""

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    """A source excerpt supporting a generated answer.

    :ivar source_id: Identifier of the source document.
    :ivar chunk_id: Identifier of the retrieved chunk within that source.
    :ivar text: Excerpt text used as supporting evidence."""

    source_id: str
    chunk_id: str
    text: str


class RetrievalMetrics(BaseModel):
    """Counts describing retrieval work for a question.

    :ivar documents_considered: Number of documents considered during retrieval.
    :ivar chunks_returned: Number of chunks returned to the generator."""

    documents_considered: int
    chunks_returned: int


class ModelMetrics(BaseModel):
    """Provider, model, and token usage for answer generation.

    Token counts are null when the provider did not report usage.

    :ivar provider: Generation provider identifier.
    :ivar name: Model identifier reported for the generation.
    :ivar input_tokens: Input token count, or ``None`` when omitted.
    :ivar output_tokens: Output token count, or ``None`` when omitted."""

    provider: str
    name: str
    input_tokens: int | None
    output_tokens: int | None


class QuestionResult(BaseModel):
    """Completed answer and supporting metadata.

    :ivar question_id: Unique identifier of the question job.
    :ivar answer: Generated answer text.
    :ivar citations: Supporting source excerpts.
    :ivar retrieval: Counts describing retrieval work.
    :ivar model: Provider, model, and token usage metadata."""

    question_id: str
    answer: str
    citations: list[Citation]
    retrieval: RetrievalMetrics
    model: ModelMetrics


class QuestionJob(BaseModel):
    """Current state of a submitted question.

    :ivar question_id: Unique identifier of the question job.
    :ivar status: Whether processing is pending, completed, or failed.
    :ivar result: Completed answer and metadata, or ``None`` when pending or failed.
    :ivar error_category: Safe failure category when ``status`` is ``failed``."""

    question_id: str
    status: Literal["pending", "completed", "failed"]
    result: QuestionResult | None = None
    error_category: Literal["processing_error"] | None = None
