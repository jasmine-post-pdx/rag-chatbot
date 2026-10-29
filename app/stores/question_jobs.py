"""Interface for question job persistence."""

from typing import Literal, Protocol

from app.models.questions import QuestionJob, QuestionResult


class IdempotencyKeyReuseError(ValueError):
    """An idempotency key was already used for a different question."""

    def __init__(self) -> None:
        super().__init__("Idempotency-Key was already used for a different question")


class QuestionJobStore(Protocol):
    """Persistence operations required by the question workflow."""

    def create_job(self, request_id: str) -> QuestionJob: ...

    def create_completed_job(
        self,
        request_id: str,
        result: QuestionResult,
        *,
        idempotency_key_hash: str | None = None,
        request_fingerprint: str | None = None,
    ) -> QuestionJob: ...

    def create_failed_job(
        self,
        request_id: str,
        error_category: Literal["processing_error"],
        *,
        idempotency_key_hash: str | None = None,
        request_fingerprint: str | None = None,
    ) -> QuestionJob: ...

    def get_job(self, request_id: str) -> QuestionJob | None: ...

    def get_by_idempotency_key(
        self, idempotency_key_hash: str
    ) -> tuple[QuestionJob, str] | None: ...

    def complete_job(self, request_id: str, result: QuestionResult) -> QuestionJob: ...

    def fail_job(
        self,
        request_id: str,
        error_category: Literal["processing_error"],
    ) -> QuestionJob: ...

    def is_ready(self) -> bool: ...

    def close(self) -> None: ...
