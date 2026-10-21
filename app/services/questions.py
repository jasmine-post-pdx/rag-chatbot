"""SQLite-backed question job storage and placeholder processing."""

import os
import sqlite3
import threading
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

    NOTE: This model is also the public response schema for
    ``GET /v1/questions/{request_id}``. Adding internal-only fields here
    will change the external API response, so consider introducing a
    separate response DTO when the job state grows beyond these fields.

    Attributes:
        request_id: Unique identifier of the question job.
        status: Whether processing is pending or completed.
        result: Completed answer, or ``None`` while processing is pending.
    """

    request_id: str
    status: Literal["pending", "completed"]
    result: QuestionResult | None = None


class QuestionService:
    """Manage question jobs and their placeholder results in SQLite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # ``check_same_thread=False`` is safe because ``self._lock`` serializes
        # access across FastAPI's thread pool workers.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the jobs table if it does not already exist."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS question_jobs (
                    request_id TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def create_job(self, request_id: str) -> QuestionJob:
        """Create and store a pending question job."""
        job = QuestionJob(request_id=request_id, status="pending")
        with self._lock:
            self._conn.execute(
                "INSERT INTO question_jobs (request_id, job_json) VALUES (?, ?)",
                (request_id, job.model_dump_json()),
            )
            self._conn.commit()
        return job

    def get_job(self, request_id: str) -> QuestionJob | None:
        """Look up a question job by identifier."""
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json FROM question_jobs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return QuestionJob.model_validate_json(row[0])

    def complete_job(self, request_id: str, question: str) -> None:
        """Complete a job with a placeholder result until RAG adapters exist."""
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
        job = QuestionJob(
            request_id=request_id,
            status="completed",
            result=result,
        )
        with self._lock:
            self._conn.execute(
                "REPLACE INTO question_jobs (request_id, job_json) VALUES (?, ?)",
                (request_id, job.model_dump_json()),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
