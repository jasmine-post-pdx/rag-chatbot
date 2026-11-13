"""Unit tests for question job storage and workflow."""

import hashlib
import sqlite3
from contextlib import closing
from uuid import UUID

import pytest

from app.domain.errors import IdempotencyKeyReuseError
from app.domain.questions import (
    ModelMetrics,
    QuestionJob,
    QuestionResult,
    RetrievalMetrics,
)
from app.services.question_workflow import (
    QuestionProcessingError,
    QuestionStatePersistenceError,
    QuestionWorkflow,
)
from app.stores.sqlite import SQLiteQuestionJobStore


def _result(question_id: str) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
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

    assert job == QuestionJob(question_id="req-1", status="pending")
    assert store.get_job("req-1") == job


def test_create_job_duplicate_question_id_raises(store_factory):
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


def test_complete_job_unknown_question_id_raises(store_factory):
    """Verify completion fails when the job does not exist."""
    store = store_factory(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        store.complete_job("missing", _result("missing"))


def test_complete_job_rejects_result_for_different_question_id(store_factory):
    """Verify mismatched results are rejected without changing the pending job."""
    store = store_factory(":memory:")
    store.create_job("req-1")

    with pytest.raises(ValueError, match="Result question ID does not match job"):
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
        question_id="req-1", status="failed", error_category="processing_error"
    )
    assert store.get_job("req-1") == failed


def test_fail_job_unknown_question_id_raises(store_factory):
    """Verify failure persistence rejects an unknown job ID."""
    store = store_factory(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        store.fail_job("missing", "processing_error")


def test_get_job_unknown_question_id_returns_none(store_factory):
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


@pytest.fixture
def legacy_question_database(tmp_path):
    """Create the original two-column table with a pre-existing serialized job."""
    db_path = tmp_path / "legacy.db"
    job_json = (
        '{"request_id":"legacy-1","status":"completed","result":'
        '{"request_id":"legacy-1","answer":"Old answer",'
        '"citations":[{"source_id":"doc","chunk_id":"chunk","text":"excerpt"}],'
        '"retrieval":{"documents_considered":1,"chunks_returned":1},'
        '"model":{"provider":"local","name":"stub","input_tokens":0,'
        '"output_tokens":0}}}'
    )
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "CREATE TABLE question_jobs (request_id TEXT PRIMARY KEY, job_json TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO question_jobs VALUES (?, ?)", ("legacy-1", job_json))
    return db_path, job_json


def test_legacy_job_remains_readable_after_upgrade_and_reopen(
    legacy_question_database, store_factory
):
    """Upgrade the old table and store the job ID as question_id."""
    db_path, job_json = legacy_question_database
    store = store_factory(str(db_path))
    job = store.get_job("legacy-1")
    assert job is not None
    assert job.status == "completed"
    assert job.error_category is None
    assert job.result is not None
    assert job.result.answer == "Old answer"
    assert job.result.citations[0].text == "excerpt"
    row = store._conn.execute(
        "SELECT job_json, created_at, idempotency_key_hash, question_fingerprint "
        "FROM question_jobs WHERE question_id = ?",
        ("legacy-1",),
    ).fetchone()
    assert row is not None
    assert row[0] == job_json.replace('"request_id":', '"question_id":')
    assert row[1]
    assert row[2:] == (None, None)
    store.create_job("new-job")
    store.close()

    reopened = store_factory(str(db_path))
    assert reopened.get_job("legacy-1") == job
    assert reopened.get_job("new-job") == QuestionJob(question_id="new-job", status="pending")


def test_store_closes_connection_if_table_setup_fails(monkeypatch):
    """A constructor failure must not leak its already-open connection."""
    conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: conn)

    def fail_setup(self):
        raise sqlite3.DatabaseError("table setup failed")

    monkeypatch.setattr(SQLiteQuestionJobStore, "_ensure_table", fail_setup)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="table setup failed"):
            SQLiteQuestionJobStore()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    finally:
        conn.close()


def test_idempotency_records_survive_reopen_and_reject_conflicts(tmp_path, store_factory):
    """Preserve keyed terminal jobs, fingerprints, and conflict semantics."""
    db_path = str(tmp_path / "questions.db")
    store = store_factory(db_path)
    job = store.create_completed_job(
        "first", _result("first"), idempotency_key_hash="key", question_fingerprint="question"
    )
    store.close()

    reopened = store_factory(db_path)
    assert reopened.get_by_idempotency_key("key") == (job, "question")
    assert reopened.create_completed_job(
        "second", _result("second"), idempotency_key_hash="key", question_fingerprint="question"
    ) == job
    with pytest.raises(IdempotencyKeyReuseError):
        reopened.create_completed_job(
            "third", _result("third"), idempotency_key_hash="key", question_fingerprint="other"
        )
    assert reopened.get_job("second") is None
    assert reopened.get_job("third") is None


# Workflow tests


def test_submission_generates_distinct_job_ids_and_forwards_correlation(store_factory):
    """The workflow assigns job IDs and forwards the HTTP correlation ID separately."""
    class RecordingProcessor:
        def __init__(self):
            self.calls = []

        def process(self, question_id, question, *, request_id=None):
            self.calls.append((question_id, question, request_id))
            return _result(question_id)

    store = store_factory(":memory:")
    processor = RecordingProcessor()
    workflow = QuestionWorkflow(store, processor)

    first = workflow.submit_question("What is RAG?", request_id="http-request")
    second = workflow.submit_question("What is RAG?", request_id="http-request")

    assert str(UUID(first.question_id, version=4)) == first.question_id
    assert str(UUID(second.question_id, version=4)) == second.question_id
    assert first.question_id != second.question_id
    assert processor.calls == [
        (first.question_id, "What is RAG?", "http-request"),
        (second.question_id, "What is RAG?", "http-request"),
    ]
    assert store.get_job(first.question_id).result == first
    assert store.get_job(second.question_id).result == second


def test_submission_reuses_idempotent_result_without_reprocessing(store_factory):
    """A repeated key returns the original job even though a new ID was generated."""
    class RecordingProcessor:
        def __init__(self):
            self.calls = 0

        def process(self, question_id, question, *, request_id=None):
            self.calls += 1
            return _result(question_id)

    processor = RecordingProcessor()
    workflow = QuestionWorkflow(store_factory(":memory:"), processor)

    first = workflow.submit_question("Question", idempotency_key=" key ")
    retry = workflow.submit_question("Question", idempotency_key="key")
    with pytest.raises(IdempotencyKeyReuseError):
        workflow.submit_question("Different", idempotency_key="key")

    assert retry == first
    assert processor.calls == 1


@pytest.mark.parametrize("job_status", ["pending", "completed"])
def test_submission_rejects_incomplete_cached_job(store_factory, job_status):
    """Incomplete saved jobs yield the same safe, pollable processing error."""
    store = store_factory(":memory:")
    store._insert_job(
        QuestionJob(question_id="existing-job", status=job_status),
        idempotency_key_hash=hashlib.sha256(b"key").hexdigest(),
        question_fingerprint=hashlib.sha256(b"Question").hexdigest(),
    )
    workflow = QuestionWorkflow(store)
    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit_question("Question", idempotency_key="key")

    assert error.value.question_id == "existing-job"
    assert error.value.exception_type == "PreviousAttemptIncomplete"


@pytest.mark.parametrize("processor_fails", [False, True])
def test_submission_reuses_winning_job_after_idempotency_race(store_factory, processor_fails):
    """A job inserted after the initial lookup wins over either processing outcome."""
    class RacingStore(SQLiteQuestionJobStore):
        def get_by_idempotency_key(self, idempotency_key_hash):
            return None

    class Processor:
        def process(self, question_id, question, *, request_id=None):
            if processor_fails:
                raise TimeoutError("provider failure")
            return _result(question_id)

    store = store_factory(store_type=RacingStore)
    original = _result("winner")
    store.create_completed_job(
        "winner", original,
        idempotency_key_hash=hashlib.sha256(b"key").hexdigest(),
        question_fingerprint=hashlib.sha256(b"Question").hexdigest(),
    )
    workflow = QuestionWorkflow(store, Processor())

    assert workflow.submit_question("Question", idempotency_key="key") == original
    assert store.get_job("winner").result == original


def test_submission_persists_failed_job_with_generated_id(store_factory):
    """The ID exposed on a processing failure identifies the persisted failed job."""
    class FailingProcessor:
        def process(self, question_id, question, *, request_id=None):
            raise TimeoutError("sensitive provider details")

    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store, FailingProcessor())
    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit_question("Question", request_id="http-request")

    assert str(UUID(error.value.question_id, version=4)) == error.value.question_id
    assert error.value.exception_type == "TimeoutError"
    assert error.value.__context__ is None
    assert store.get_job(error.value.question_id).status == "failed"


def test_workflow_completes_job_with_placeholder_result(store_factory):
    """Verify the default processor completes a job with its placeholder result."""
    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store, question_id_factory=lambda: "req-1")

    result = workflow.submit_question("What is RAG?")

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
            question_id: str,
            question: str,
            *,
            request_id: str | None = None,
        ) -> QuestionResult:
            self.call = (question_id, question)
            return expected

    store = store_factory(":memory:")
    processor = RecordingProcessor()
    workflow = QuestionWorkflow(store, processor, question_id_factory=lambda: "req-custom")

    result = workflow.submit_question("Use my processor")

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
            question_id: str,
            question: str,
            *,
            request_id: str | None = None,
        ) -> QuestionResult:
            raise TimeoutError("private provider endpoint and credential detail")

    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store, FailingProcessor(), question_id_factory=lambda: "req-1")

    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit_question("What is RAG?")

    assert error.value.exception_type == "TimeoutError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None

    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
    assert job.result is None


def test_workflow_wraps_completed_job_persistence_failure(store_factory):
    """Completed-job persistence failures become QuestionStatePersistenceError."""
    class FailingCompletedStore(SQLiteQuestionJobStore):
        def create_completed_job(self, question_id, result, **kwargs):
            raise RuntimeError("database connection detail must not leak")

    store = store_factory(store_type=FailingCompletedStore)
    workflow = QuestionWorkflow(store, question_id_factory=lambda: "req-1")

    with pytest.raises(QuestionStatePersistenceError) as error:
        workflow.submit_question("What is RAG?")

    assert error.value.question_id == "req-1"
    assert error.value.exception_type == "RuntimeError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert store.get_job("req-1") is None


def test_workflow_marks_mismatched_processor_result_as_failed(store_factory):
    """Verify a processor result for another job ID becomes a processing failure."""
    class MismatchedProcessor:
        def process(
            self,
            question_id: str,
            question: str,
            *,
            request_id: str | None = None,
        ) -> QuestionResult:
            return _result("different-request")

    store = store_factory(":memory:")
    workflow = QuestionWorkflow(store, MismatchedProcessor(), question_id_factory=lambda: "req-1")

    with pytest.raises(QuestionProcessingError) as error:
        workflow.submit_question("What is RAG?")

    assert error.value.exception_type == "ValueError"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    job = store.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
