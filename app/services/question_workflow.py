"""Coordinate question processing with question job persistence."""

import hashlib

from app.models.questions import QuestionJob, QuestionResult
from app.services.question_processor import (
    PlaceholderQuestionProcessor,
    QuestionProcessor,
)
from app.stores.question_jobs import IdempotencyKeyReuseError, QuestionJobStore


class QuestionProcessingError(Exception):
    """A processing failure whose persisted job can be retrieved by ID."""

    def __init__(self, request_id: str, exception_type: str) -> None:
        super().__init__("Question processing failed")
        self.request_id = request_id
        self.exception_type = exception_type


class QuestionStatePersistenceError(Exception):
    """A processing failure whose failed state could not be persisted."""

    def __init__(self, request_id: str, exception_type: str) -> None:
        super().__init__("Question job failure state could not be persisted")
        self.request_id = request_id
        self.exception_type = exception_type


class QuestionWorkflow:
    """Process questions and persist only terminal job states."""

    def __init__(self, store: QuestionJobStore, processor: QuestionProcessor | None = None):
        self._store = store
        self._processor = processor or PlaceholderQuestionProcessor()

    def submit(
        self,
        request_id: str,
        question: str,
        *,
        correlation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> QuestionResult:
        """Process a question, then persist either success or failure.

        Args:
            request_id: Identifier assigned to the question job.
            question: User question to process.
            correlation_id: Optional middleware request ID for log correlation.
            idempotency_key: Optional client key used to reuse a terminal result.

        Returns:
            The generated and persisted question result.
        """
        idempotency_key_hash: str | None = None
        request_fingerprint: str | None = None
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 255:
                raise ValueError("Idempotency-Key must contain 1 to 255 characters")
            idempotency_key_hash = hashlib.sha256(
                idempotency_key.encode("utf-8")
            ).hexdigest()
            request_fingerprint = hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest()
            existing = self._store.get_by_idempotency_key(idempotency_key_hash)
            if existing is not None:
                existing_job, existing_fingerprint = existing
                if existing_fingerprint != request_fingerprint:
                    raise IdempotencyKeyReuseError() from None
                return _result_for_terminal_job(existing_job)

        processing_exception_type: str | None = None
        persistence_exception_type: str | None = None
        idempotency_conflict = False
        terminal_job: QuestionJob | None = None
        result: QuestionResult | None = None
        try:
            result = self._processor.process(
                request_id,
                question,
                correlation_id=correlation_id,
            )
            if result.request_id != request_id:
                raise ValueError(
                    f"Processor returned result for a different request: {result.request_id}"
                )
        except Exception as exc:
            processing_exception_type = type(exc).__name__
            try:
                terminal_job = self._store.create_failed_job(
                    request_id,
                    "processing_error",
                    idempotency_key_hash=idempotency_key_hash,
                    request_fingerprint=request_fingerprint,
                )
            except IdempotencyKeyReuseError:
                idempotency_conflict = True
            except Exception as persistence_exc:
                persistence_exception_type = type(persistence_exc).__name__

        # Raise after leaving the except blocks so the original exception objects
        # are not retained as __cause__ or __context__ on the public wrapper.
        if idempotency_conflict:
            raise IdempotencyKeyReuseError() from None
        if persistence_exception_type is not None:
            raise QuestionStatePersistenceError(
                request_id, persistence_exception_type
            ) from None
        if processing_exception_type is not None:
            if terminal_job is not None and terminal_job.request_id != request_id:
                return _result_for_terminal_job(terminal_job)
            raise QuestionProcessingError(
                request_id, processing_exception_type
            ) from None
        if result is None:
            raise RuntimeError("Question processor returned no result")
        terminal_job = self._store.create_completed_job(
            request_id,
            result,
            idempotency_key_hash=idempotency_key_hash,
            request_fingerprint=request_fingerprint,
        )
        if terminal_job.request_id != request_id:
            return _result_for_terminal_job(terminal_job)
        return result

    def get_job(self, request_id: str) -> QuestionJob | None:
        """Return the persisted job state."""
        return self._store.get_job(request_id)

    def is_ready(self) -> bool:
        """Delegate readiness checks to the configured store."""
        return self._store.is_ready()

    def close(self) -> None:
        """Close the configured job store."""
        self._store.close()


def _result_for_terminal_job(job: QuestionJob) -> QuestionResult:
    """Return a saved result or reproduce the generic response for a prior failure."""
    if job.status == "completed" and job.result is not None:
        return job.result
    if job.status == "failed":
        raise QuestionProcessingError(job.request_id, "PreviousAttemptFailed") from None
    raise QuestionProcessingError(job.request_id, "PreviousAttemptIncomplete") from None
