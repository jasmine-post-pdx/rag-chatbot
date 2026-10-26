"""SQLite implementation of question job persistence."""

import os
import sqlite3
import threading
from typing import Literal

from app.models.questions import QuestionJob, QuestionResult


class SQLiteQuestionJobStore:
    """Persist question jobs as JSON records in SQLite."""

    def __init__(self, db_path: str = ":memory:", timeout: float = 5.0) -> None:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
        self._lock = threading.Lock()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the jobs table and enable WAL for file-backed databases."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS question_jobs (
                    request_id TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def create_job(self, request_id: str) -> QuestionJob:
        """Create and store a pending job."""
        job = QuestionJob(request_id=request_id, status="pending")
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO question_jobs (request_id, job_json) VALUES (?, ?)",
                    (request_id, job.model_dump_json()),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Job already exists: {request_id}") from exc
        return job

    def get_job(self, request_id: str) -> QuestionJob | None:
        """Load a job, returning None when it does not exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json FROM question_jobs WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        return QuestionJob.model_validate_json(row[0])

    def complete_job(self, request_id: str, result: QuestionResult) -> QuestionJob:
        """Persist a completed result for an existing job."""
        if result.request_id != request_id:
            raise ValueError(
                f"Result request ID does not match job: {result.request_id} != {request_id}"
            )
        job = QuestionJob(request_id=request_id, status="completed", result=result)
        self._save_job(job)
        return job

    def fail_job(
        self, request_id: str, error_category: Literal["processing_error"]
    ) -> QuestionJob:
        """Persist a safe failure category for an existing job."""
        job = QuestionJob(
            request_id=request_id,
            status="failed",
            error_category=error_category,
        )
        self._save_job(job)
        return job

    def _save_job(self, job: QuestionJob) -> None:
        """Replace an existing job record."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE question_jobs SET job_json = ? WHERE request_id = ?",
                (job.model_dump_json(), job.request_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Job not found: {job.request_id}")
            self._conn.commit()

    def is_ready(self) -> bool:
        """Return whether a simple query succeeds."""
        try:
            with self._lock:
                self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()
