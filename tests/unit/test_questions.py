"""Unit tests for the question job service."""

import pytest

from app.services.questions import QuestionJob, QuestionService


def test_create_job_stores_pending_job():
    """Verify create_job persists a pending job."""
    service = QuestionService(":memory:")

    job = service.create_job("req-1")

    assert job == QuestionJob(request_id="req-1", status="pending")
    retrieved = service.get_job("req-1")
    assert retrieved is not None
    assert retrieved.status == "pending"


def test_create_job_duplicate_request_id_raises():
    """Verify create_job fails when the request ID already exists."""
    service = QuestionService(":memory:")
    service.create_job("req-1")

    with pytest.raises(ValueError, match="Job already exists: req-1"):
        service.create_job("req-1")


def test_complete_job_updates_existing_job():
    """Verify complete_job transitions a pending job to completed."""
    service = QuestionService(":memory:")
    service.create_job("req-1")

    service.complete_job("req-1", "What is RAG?")

    job = service.get_job("req-1")
    assert job is not None
    assert job.status == "completed"
    assert job.result is not None
    assert job.result.request_id == "req-1"
    assert "What is RAG?" in job.result.answer


def test_complete_job_unknown_request_id_raises():
    """Verify complete_job fails when the job does not exist."""
    service = QuestionService(":memory:")

    with pytest.raises(ValueError, match="Job not found: missing"):
        service.complete_job("missing", "What is RAG?")


def test_get_job_unknown_request_id_returns_none():
    """Verify get_job returns None for an unknown request ID."""
    service = QuestionService(":memory:")

    assert service.get_job("missing") is None


def test_is_ready_true_for_healthy_database():
    """Verify is_ready reports True for a usable connection."""
    service = QuestionService(":memory:")

    assert service.is_ready() is True


def test_close_makes_database_unready():
    """Verify is_ready reports False after the connection is closed."""
    service = QuestionService(":memory:")
    service.close()

    assert service.is_ready() is False


def test_file_backed_database_uses_wal_mode(tmp_path):
    """Verify file-backed databases are created with WAL journal mode."""
    db_path = tmp_path / "questions.db"
    service = QuestionService(str(db_path))

    row = service._conn.execute("PRAGMA journal_mode").fetchone()
    assert row is not None
    assert row[0] == "wal"


def test_database_timeout_is_configurable():
    """Verify the SQLite connection accepts a custom timeout parameter."""
    service = QuestionService(":memory:", timeout=10.0)

    assert service.is_ready() is True
