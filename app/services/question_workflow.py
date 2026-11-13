"""Coordinate question processing with question job persistence."""

import hashlib
from collections.abc import Callable
from uuid import uuid4

from app.domain.errors import (
    IdempotencyKeyReuseError,
    IncompleteIdempotencyRecordError,
    InvalidIdempotencyKeyError,
)
from app.domain.questions import QuestionJob, QuestionResult
from app.services.question_processor import (
    PlaceholderQuestionProcessor,
    QuestionProcessor,
)
from app.stores.question_job_store import QuestionJobStore


class QuestionProcessingError(Exception):
    """A processing failure whose persisted job can be retrieved by ID."""

    def __init__(self, question_id: str, exception_type: str) -> None:
        """Record the failed job ID and underlying exception type name.

        :param question_id: Identifier of the persisted failed job.
        :param exception_type: Short name of the exception that caused failure."""
        super().__init__("Question processing failed")
        self.question_id = question_id
        self.exception_type = exception_type


class QuestionStatePersistenceError(Exception):
    """A terminal job state could not be persisted after processing."""

    def __init__(self, question_id: str, exception_type: str) -> None:
        """Record the job ID and persistence exception type name.

        :param question_id: Identifier of the job that could not be saved.
        :param exception_type: Short name of the persistence exception."""
        super().__init__("Question job state could not be persisted")
        self.question_id = question_id
        self.exception_type = exception_type


class QuestionWorkflow:
    """Process questions and persist only terminal job states."""

    def __init__(
        self,
        store: QuestionJobStore,
        processor: QuestionProcessor | None = None,
        question_id_factory: Callable[[], str] = lambda: str(uuid4()),
    ):
        """Create a workflow that coordinates processing and persistence.

        :param store: Persistence backend for terminal question jobs.
        :param processor: Answer processor. Defaults to the placeholder processor.
        :param question_id_factory: Callable that returns a new question ID."""
        self._store = store
        self._processor = processor or PlaceholderQuestionProcessor()
        self._question_id_factory = question_id_factory

    def submit_question(
        self,
        question: str,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> QuestionResult:
        """Assign a question ID and process the question synchronously.

        ``request_id`` is the HTTP correlation ID. It is independent of the
        question ID. An idempotency key can reuse a previously persisted job.

        :param question: User question to process.
        :param request_id: Optional HTTP correlation ID for log correlation.
        :param idempotency_key: Optional client key used to reuse a terminal result.
        :returns: The generated and persisted question result."""
        return self._process_and_persist(
            self._question_id_factory(),
            question,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )

    def _process_and_persist(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> QuestionResult:
        """Process a question, then persist either success or failure.

        :param question_id: Service-generated identifier assigned to the question job.
        :param question: User question to process.
        :param request_id: Optional HTTP correlation ID for log correlation.
        :param idempotency_key: Optional client key used to reuse a terminal result.
        :returns: The generated and persisted question result."""
        idempotency_key_hash: str | None = None
        question_fingerprint: str | None = None
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 255:
                raise InvalidIdempotencyKeyError() from None
            idempotency_key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
            question_fingerprint = hashlib.sha256(question.encode("utf-8")).hexdigest()
            existing = self._store.get_by_idempotency_key(idempotency_key_hash)
            if existing is not None:
                existing_job, existing_fingerprint = existing
                if existing_fingerprint != question_fingerprint:
                    raise IdempotencyKeyReuseError() from None
                return _result_for_terminal_job(existing_job)

        processing_exception_type: str | None = None
        persistence_exception_type: str | None = None
        idempotency_conflict = False
        incomplete_idempotency_record = False
        terminal_job: QuestionJob | None = None
        result: QuestionResult | None = None
        try:
            result = self._processor.process(
                question_id,
                question,
                request_id=request_id,
            )
            if result.question_id != question_id:
                raise ValueError(
                    f"Processor returned result for a different question: {result.question_id}"
                )
        except Exception as exc:
            processing_exception_type = type(exc).__name__
            try:
                terminal_job = self._store.create_failed_job(
                    question_id,
                    "processing_error",
                    idempotency_key_hash=idempotency_key_hash,
                    question_fingerprint=question_fingerprint,
                )
            except IdempotencyKeyReuseError:
                idempotency_conflict = True
            except IncompleteIdempotencyRecordError:
                incomplete_idempotency_record = True
            except Exception as persistence_exc:
                persistence_exception_type = type(persistence_exc).__name__
        else:
            try:
                terminal_job = self._store.create_completed_job(
                    question_id,
                    result,
                    idempotency_key_hash=idempotency_key_hash,
                    question_fingerprint=question_fingerprint,
                )
            except IdempotencyKeyReuseError:
                idempotency_conflict = True
            except IncompleteIdempotencyRecordError:
                incomplete_idempotency_record = True
            except Exception as persistence_exc:
                persistence_exception_type = type(persistence_exc).__name__

        # Raise after leaving the except blocks so the original exception objects
        # are not retained as __cause__ or __context__ on the public wrapper.
        if incomplete_idempotency_record:
            raise IncompleteIdempotencyRecordError() from None
        if idempotency_conflict:
            raise IdempotencyKeyReuseError() from None
        if persistence_exception_type is not None:
            raise QuestionStatePersistenceError(question_id, persistence_exception_type) from None
        if processing_exception_type is not None:
            if terminal_job is not None and terminal_job.question_id != question_id:
                return _result_for_terminal_job(terminal_job)
            raise QuestionProcessingError(question_id, processing_exception_type) from None
        if result is None or terminal_job is None:
            raise RuntimeError("Question processor returned no result")
        if terminal_job.question_id != question_id:
            return _result_for_terminal_job(terminal_job)
        return result

    def get_job(self, question_id: str) -> QuestionJob | None:
        """Return the persisted job state.

        :param question_id: Unique identifier of the job to load.
        :returns: The stored job, or ``None`` when it does not exist."""
        return self._store.get_job(question_id)

    def close(self) -> None:
        """Close the configured job store."""
        self._store.close()


def _result_for_terminal_job(job: QuestionJob) -> QuestionResult:
    """Return a saved result or reproduce the generic response for a prior failure.

    :param job: Terminal job loaded from the store.
    :returns: The completed result when the job finished successfully.
    :raises QuestionProcessingError: If the prior attempt failed or is incomplete."""
    if job.status == "completed" and job.result is not None:
        return job.result
    if job.status == "failed":
        raise QuestionProcessingError(job.question_id, "PreviousAttemptFailed") from None
    raise QuestionProcessingError(job.question_id, "PreviousAttemptIncomplete") from None
