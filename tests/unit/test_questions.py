"""Unit tests for question job storage and workflow."""

import logging

import pytest

from app.models.questions import (
    ModelMetrics,
    QuestionJob,
    QuestionResult,
    RetrievalMetrics,
)
from app.services.question_workflow import QuestionProcessingError, QuestionWorkflow
from app.stores.sqlite_question_jobs import SQLiteQuestionJobStore


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


def test_create_job_stores_pending_job():
    store = SQLiteQuestionJobStore(":memory:")

    job = store.create_job("req-1")

    assert job == QuestionJob(request_id="req-1", status="pending")
    assert store.get_job("req-1") == job


def test_create_job_duplicate_request_id_raises():
    store = SQLiteQuestionJobStore(":memory:")
    store.create_job("req-1")

    with pytest.raises(ValueError, match="Job already exists: req-1"):
        store.create_job("req-1")


def test_complete_job_persists_supplied_result():
    store = SQLiteQuestionJobStore(":memory:")
    store.create_job("req-1")

    completed = store.complete_job("req-1", _result("req-1"))

    assert completed.status == "completed"
    assert completed.result == _result("req-1")
    assert store.get_job("req-1") == completed


def test_complete_job_unknown_request_id_raises():
    store = SQLiteQuestionJobStore(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        store.complete_job("missing", _result("missing"))


def test_complete_job_rejects_result_for_different_request_id():
    store = SQLiteQuestionJobStore(":memory:")
    store.create_job("req-1")

    with pytest.raises(ValueError, match="Result request ID does not match job"):
        store.complete_job("req-1", _result("req-2"))


def test_fail_job_persists_safe_error_category():
    store = SQLiteQuestionJobStore(":memory:")
    store.create_job("req-1")

    failed = store.fail_job("req-1", "processing_error")

    assert failed == QuestionJob(
        request_id="req-1", status="failed", error_category="processing_error"
    )
    assert store.get_job("req-1") == failed


def test_fail_job_unknown_request_id_raises():
    store = SQLiteQuestionJobStore(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        store.fail_job("missing", "processing_error")


def test_workflow_completes_job_with_placeholder_result():
    store = SQLiteQuestionJobStore(":memory:")
    workflow = QuestionWorkflow(store)

    result = workflow.submit("req-1", "What is RAG?")

    assert result.answer == "This is a placeholder answer for: What is RAG?"
    job = workflow.get_job("req-1")
    assert job is not None
    assert job.status == "completed"


def test_workflow_passes_question_to_processor_and_persists_its_result():
    expected = _result("req-custom")

    class RecordingProcessor:
        def __init__(self):
            self.call = None

        def process(self, request_id: str, question: str) -> QuestionResult:
            self.call = (request_id, question)
            return expected

    store = SQLiteQuestionJobStore(":memory:")
    processor = RecordingProcessor()
    workflow = QuestionWorkflow(store, processor)

    result = workflow.submit("req-custom", "Use my processor")

    assert processor.call == ("req-custom", "Use my processor")
    assert result == expected
    job = store.get_job("req-custom")
    assert job is not None
    assert job.status == "completed"
    assert job.result == expected


def test_workflow_persists_failure_and_logs_safe_processor_diagnostic(caplog):
    class FailingProcessor:
        def process(self, request_id: str, question: str) -> QuestionResult:
            raise TimeoutError("private provider endpoint and credential detail")

    store = SQLiteQuestionJobStore(":memory:")
    workflow = QuestionWorkflow(store, FailingProcessor())
    caplog.set_level(logging.ERROR, logger="app.services.question_workflow")

    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit("req-1", "What is RAG?")

    assert isinstance(error.value.__cause__, TimeoutError)
    assert "private provider endpoint" in str(error.value.__cause__)
    assert "TimeoutError" in caplog.text
    assert "credential detail" not in caplog.text

    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
    assert job.result is None


def test_workflow_marks_mismatched_processor_result_as_failed():
    class MismatchedProcessor:
        def process(self, request_id: str, question: str) -> QuestionResult:
            return _result("different-request")

    store = SQLiteQuestionJobStore(":memory:")
    workflow = QuestionWorkflow(store, MismatchedProcessor())

    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit("req-1", "What is RAG?")

    assert isinstance(error.value.__cause__, ValueError)
    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"


def test_get_job_unknown_request_id_returns_none():
    store = SQLiteQuestionJobStore(":memory:")

    assert store.get_job("missing") is None


def test_is_ready_true_for_healthy_database():
    store = SQLiteQuestionJobStore(":memory:")

    assert store.is_ready() is True


def test_close_makes_database_unready():
    store = SQLiteQuestionJobStore(":memory:")
    store.close()

    assert store.is_ready() is False


def test_file_backed_database_uses_wal_mode(tmp_path):
    store = SQLiteQuestionJobStore(str(tmp_path / "questions.db"))

    row = store._conn.execute("PRAGMA journal_mode").fetchone()
    assert row is not None
    assert row[0] == "wal"


def test_completed_and_failed_jobs_survive_store_reopen(tmp_path):
    db_path = str(tmp_path / "questions.db")
    first_store = SQLiteQuestionJobStore(db_path)
    first_store.create_job("completed-job")
    completed = _result("completed-job")
    first_store.complete_job("completed-job", completed)
    first_store.create_job("failed-job")
    first_store.fail_job("failed-job", "processing_error")
    first_store.close()

    reopened_store = SQLiteQuestionJobStore(db_path)

    assert reopened_store.get_job("completed-job").result == completed
    failed = reopened_store.get_job("failed-job")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_category == "processing_error"


def test_database_timeout_is_configurable():
    store = SQLiteQuestionJobStore(":memory:", timeout=10.0)

    assert store.is_ready() is True
