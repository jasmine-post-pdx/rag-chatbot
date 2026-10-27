"""Coordinate question processing with question job persistence."""

from app.models.questions import QuestionJob, QuestionResult
from app.services.question_processor import (
    PlaceholderQuestionProcessor,
    QuestionProcessor,
)
from app.stores.question_jobs import QuestionJobStore


class QuestionProcessingError(Exception):
    """A processing failure whose persisted job can be retrieved by ID."""

    def __init__(self, request_id: str) -> None:
        super().__init__("Question processing failed")
        self.request_id = request_id


class QuestionWorkflow:
    """Orchestrate job creation, answer processing, and result persistence."""

    def __init__(self, store: QuestionJobStore, processor: QuestionProcessor | None = None):
        self._store = store
        self._processor = processor or PlaceholderQuestionProcessor()

    def submit(self, request_id: str, question: str) -> QuestionResult:
        """Create a job, process it, and persist either success or failure."""
        self._store.create_job(request_id)
        try:
            result = self._processor.process(request_id, question)
            if result.request_id != request_id:
                raise ValueError(
                    f"Processor returned result for a different request: {result.request_id}"
                )
        except Exception as exc:
            self._store.fail_job(request_id, "processing_error")
            raise QuestionProcessingError(request_id) from exc
        self._store.complete_job(request_id, result)
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
