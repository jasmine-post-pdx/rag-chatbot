"""Unit tests for question job storage and workflow."""

import pytest

from app.models.questions import (
    ModelMetrics,
    QuestionJob,
    QuestionResult,
    RetrievalMetrics,
)
from app.services.question_workflow import QuestionProcessingError, QuestionWorkflow


def _result(request_id: str) -> QuestionResult:
    return QuestionResult(
        request_id=request_id,
        answer="A test answer",
        citations=[],
        retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
        model=ModelMetrics(
            provider="test", name="test", input_tokens=0, output_tokens=0
        ),
    )


# SQLite job store tests


def test_create_job_stores_pending_job(store_factory):
    """Verify job creation persists a pending job."""
    store = store_factory(":memory:")

    job = store.create_job("req-1")

    assert job == QuestionJob(request_id="req-1", status="pending")
    assert store.get_job("req-1") == job


def test_create_job_duplicate_request_id_raises(store_factory):
    """Verify creating a duplicate job raises a value error."""
    store = store_factory(":memory:")
    store.create_job("req-1")

    with pytest.raises(ValueError, match="Job already exists: req-1"):
        store.create_job("req-1")


def test_complete_job_persists_supplied_result(store_factory):
    """Verify completion persists the supplied result."""
    store = store_factory(":memory:")
    store.create_job("req-1")

    completed = store.complete_job("req-1", _result("req-1"))

    assert completed.status == "completed"
    assert completed.result == _result("req-1")
    assert store.get_job("req-1") == completed


def test_complete_job_unknown_request_id_raises(store_factory):
    """Verify completion fails when the job does not exist."""
    store = store_factory(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        store.complete_job("missing", _result("missing"))


def test_complete_job_rejects_result_for_different_request_id(store_factory):
    """Verify mismatched results are rejected without changing the pending job."""
    store = store_factory(":memory:")
    store.create_job("req-1")

    with pytest.raises(ValueError, match="Result request ID does not match job"):
        store.complete_job("req-1", _result("req-2"))

    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "pending"
    assert job.result is None


def test_fail_job_persists_safe_error_category(store_factory):
    """Verify failure persists the safe error category."""
    store = store_factory(":memory:")
    store.create_job("req-1")

    failed = store.fail_job("req-1", "processing_error")

    assert failed == QuestionJob(
        request_id="req-1", status="failed", error_category="processing_error"
    )
    assert store.get_job("req-1") == failed


def test_fail_job_unknown_request_id_raises(store_factory):
    """Verify failure persistence rejects an unknown job ID."""
    store = store_factory(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        store.fail_job("missing", "processing_error")


def test_get_job_unknown_request_id_returns_none(store_factory):
    """Verify looking up an unknown job returns None."""
    store = store_factory(":memory:")

    assert store.get_job("missing") is None


def test_is_ready_true_for_healthy_database(store_factory):
    """Verify readiness returns true for a usable database."""
    store = store_factory(":memory:")

    assert store.is_ready() is True


def test_close_makes_database_unready(store_factory):
    """Verify closing the store makes readiness checks fail."""
    store = store_factory(":memory:")
    store.close()

    assert store.is_ready() is False


def test_file_backed_database_uses_wal_mode(tmp_path, store_factory):
    """Verify file-backed databases use WAL journal mode."""
    store = store_factory(str(tmp_path / "questions.db"))

    row = store._conn.execute("PRAGMA journal_mode").fetchone()
    assert row is not None
    assert row[0] == "wal"


def test_completed_and_failed_jobs_survive_store_reopen(tmp_path, store_factory):
    """Verify completed and failed jobs persist after reopening the store."""
    db_path = str(tmp_path / "questions.db")
    first_store = store_factory(db_path)
    first_store.create_job("completed-job")
    completed = _result("completed-job")
    first_store.complete_job("completed-job", completed)
    first_store.create_job("failed-job")
    first_store.fail_job("failed-job", "processing_error")
    first_store.close()

    reopened_store = store_factory(db_path)

    completed_job = reopened_store.get_job("completed-job")
    assert completed_job is not None
    assert completed_job.result == completed
    failed = reopened_store.get_job("failed-job")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_category == "processing_error"


def test_database_timeout_is_configurable(store_factory):
    """Verify the configured timeout is applied to SQLite busy waits."""
    store = store_factory(":memory:", timeout=10.0)

    row = store._conn.execute("PRAGMA busy_timeout").fetchone()
    assert row is not None
    assert row[0] == 10_000


# Workflow tests


def test_workflow_completes_job_with_placeholder_result(store_factory):
    """Verify the default processor completes a job with its placeholder result."""
    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store)

    result = workflow.submit("req-1", "What is RAG?")

    assert result.answer == "This is a placeholder answer for: What is RAG?"
    job = workflow.get_job("req-1")
    assert job is not None
    assert job.status == "completed"


def test_workflow_passes_question_to_processor_and_persists_its_result(store_factory):
    """Verify the workflow forwards inputs and persists the processor result."""
    expected = _result("req-custom")

    class RecordingProcessor:
        def __init__(self):
            self.call = None

        def process(
            self,
            request_id: str,
            question: str,
            *,
            correlation_id: str | None = None,
        ) -> QuestionResult:
            self.call = (request_id, question)
            return expected

    store = store_factory(":memory:")
    processor = RecordingProcessor()
    workflow = QuestionWorkflow(store, processor)

    result = workflow.submit("req-custom", "Use my processor")

    assert processor.call == ("req-custom", "Use my processor")
    assert result == expected
    job = store.get_job("req-custom")
    assert job is not None
    assert job.status == "completed"
    assert job.result == expected


def test_workflow_persists_failure_without_preserving_processor_exception(store_factory):
    """Persist a failed job and expose only safe processor exception metadata."""
    class FailingProcessor:
        def process(
            self,
            request_id: str,
            question: str,
            *,
            correlation_id: str | None = None,
        ) -> QuestionResult:
            raise TimeoutError("private provider endpoint and credential detail")

    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store, FailingProcessor())

    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit("req-1", "What is RAG?")

    assert error.value.exception_type == "TimeoutError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None

    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
    assert job.result is None


def test_workflow_marks_mismatched_processor_result_as_failed(store_factory):
    """Verify a processor result for another job ID becomes a processing failure."""
    class MismatchedProcessor:
        def process(
            self,
            request_id: str,
            question: str,
            *,
            correlation_id: str | None = None,
        ) -> QuestionResult:
            return _result("different-request")

    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store, MismatchedProcessor())

    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit("req-1", "What is RAG?")

    assert error.value.exception_type == "ValueError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
