"""Interface for question job persistence."""

from typing import Literal, Protocol

from app.models.questions import QuestionJob, QuestionResult


class QuestionJobStore(Protocol):
    """Persistence operations required by the question workflow."""

    def create_job(self, request_id: str) -> QuestionJob: ...

    def get_job(self, request_id: str) -> QuestionJob | None: ...

    def complete_job(self, request_id: str, result: QuestionResult) -> QuestionJob: ...

    def fail_job(
        self,
        request_id: str,
        error_category: Literal["processing_error"],
    ) -> QuestionJob: ...

    def is_ready(self) -> bool: ...

    def close(self) -> None: ...
