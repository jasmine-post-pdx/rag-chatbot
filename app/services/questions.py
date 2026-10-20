"""In-memory question job storage and placeholder processing."""

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    """A source excerpt supporting a generated answer.

    Attributes:
        source_id: Identifier of the source document.
        chunk_id: Identifier of the cited chunk.
        text: Supporting excerpt from the source.
    """

    source_id: str
    chunk_id: str
    text: str


class RetrievalMetrics(BaseModel):
    """Counts describing the retrieval work for a question.

    Attributes:
        documents_considered: Number of documents examined.
        chunks_returned: Number of chunks returned to generation.
    """

    documents_considered: int
    chunks_returned: int


class ModelMetrics(BaseModel):
    """Provider, model, and token usage for answer generation.

    Attributes:
        provider: Provider category or name.
        name: Model name.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens produced.
    """

    provider: str
    name: str
    input_tokens: int
    output_tokens: int


class QuestionResult(BaseModel):
    """Completed answer and its supporting metadata.

    Attributes:
        request_id: Unique identifier of the question job.
        answer: Generated answer text.
        citations: Source excerpts supporting the answer.
        retrieval: Retrieval counts for the job.
        model: Model and token usage metadata.
    """

    request_id: str
    answer: str
    citations: list[Citation]
    retrieval: RetrievalMetrics
    model: ModelMetrics


class QuestionJob(BaseModel):
    """Current state of a submitted question.

    Attributes:
        request_id: Unique identifier of the question job.
        status: Whether processing is pending or completed.
        result: Completed answer, or ``None`` while processing is pending.
    """

    request_id: str
    status: Literal["pending", "completed"]
    result: QuestionResult | None = None


_jobs: dict[str, QuestionJob] = {}


def create_job(request_id: str) -> QuestionJob:
    """Create and store a pending question job.

    Args:
        request_id: Unique identifier assigned to the job.

    Returns:
        The newly created pending job.
    """
    job = QuestionJob(request_id=request_id, status="pending")
    _jobs[request_id] = job
    return job


def get_job(request_id: str) -> QuestionJob | None:
    """Look up a question job by identifier.

    Args:
        request_id: Unique identifier of the job to retrieve.

    Returns:
        The stored job, or ``None`` if it does not exist.
    """
    return _jobs.get(request_id)


def complete_job(request_id: str, question: str) -> None:
    """Complete a job with a placeholder result until RAG adapters exist.

    Args:
        request_id: Unique identifier of the job to complete.
        question: Original question text used in the placeholder answer.
    """
    result = QuestionResult(
        request_id=request_id,
        answer=f"This is a placeholder answer for: {question}",
        citations=[],
        retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
        model=ModelMetrics(
            provider="local-or-api",
            name="stub",
            input_tokens=0,
            output_tokens=0,
        ),
    )
    _jobs[request_id] = QuestionJob(
        request_id=request_id,
        status="completed",
        result=result,
    )
