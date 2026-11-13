"""Interface for question job persistence."""

from typing import Literal, Protocol

from app.domain.questions import QuestionJob, QuestionResult


class QuestionJobStore(Protocol):
    """Persistence operations required by the question workflow."""

    def create_job(self, question_id: str) -> QuestionJob:
        """Create and store a pending job.

        :param question_id: Unique identifier for the new job.
        :returns: The created pending job."""
        ...

    def create_completed_job(
        self,
        question_id: str,
        result: QuestionResult,
        *,
        idempotency_key_hash: str | None = None,
        question_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a completed job without an intermediate pending state.

        :param question_id: Unique identifier for the job.
        :param result: Completed answer and metadata to persist.
        :param idempotency_key_hash: Optional SHA-256 hash of the client key.
        :param question_fingerprint: Optional SHA-256 hash of the question text.
        :returns: The persisted completed job, or an existing matching terminal job
            when an idempotency conflict resolves to the same fingerprint."""
        ...

    def create_failed_job(
        self,
        question_id: str,
        error_category: Literal["processing_error"],
        *,
        idempotency_key_hash: str | None = None,
        question_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a failed job without an intermediate pending state.

        :param question_id: Unique identifier for the job.
        :param error_category: Safe failure category stored on the job.
        :param idempotency_key_hash: Optional SHA-256 hash of the client key.
        :param question_fingerprint: Optional SHA-256 hash of the question text.
        :returns: The persisted failed job, or an existing matching terminal job
            when an idempotency conflict resolves to the same fingerprint."""
        ...

    def get_job(self, question_id: str) -> QuestionJob | None:
        """Load a job by identifier.

        :param question_id: Unique identifier of the job to load.
        :returns: The stored job, or ``None`` when it does not exist."""
        ...

    def get_by_idempotency_key(
        self, idempotency_key_hash: str
    ) -> tuple[QuestionJob, str] | None:
        """Load a job and its question fingerprint by hashed idempotency key.

        :param idempotency_key_hash: SHA-256 hash of the client idempotency key.
        :returns: A ``(job, question_fingerprint)`` pair, or ``None`` when no row
            exists for the key."""
        ...

    def complete_job(self, question_id: str, result: QuestionResult) -> QuestionJob:
        """Persist a completed result for an existing job.

        :param question_id: Unique identifier of the job to update.
        :param result: Completed answer and metadata to store.
        :returns: The updated completed job."""
        ...

    def fail_job(
        self,
        question_id: str,
        error_category: Literal["processing_error"],
    ) -> QuestionJob:
        """Persist a safe failure category for an existing job.

        :param question_id: Unique identifier of the job to update.
        :param error_category: Safe failure category stored on the job.
        :returns: The updated failed job."""
        ...

    def is_ready(self) -> bool:
        """Return whether the store can answer readiness checks.

        :returns: ``True`` when the backing store is queryable."""
        ...

    def close(self) -> None:
        """Release store resources such as database connections."""
        ...
