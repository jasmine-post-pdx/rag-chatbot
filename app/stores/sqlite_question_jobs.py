"""SQLite implementation of question job persistence."""

import os
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Literal

from app.models.questions import QuestionJob, QuestionResult
from app.stores.question_jobs import IdempotencyKeyReuseError


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
                    job_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    idempotency_key_hash TEXT,
                    request_fingerprint TEXT
                )
                """
            )
            columns = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(question_jobs)").fetchall()
            }
            if "idempotency_key_hash" not in columns:
                self._conn.execute(
                    "ALTER TABLE question_jobs ADD COLUMN idempotency_key_hash TEXT"
                )
            if "created_at" not in columns:
                self._conn.execute(
                    "ALTER TABLE question_jobs ADD COLUMN created_at TEXT"
                )
            if "request_fingerprint" not in columns:
                self._conn.execute(
                    "ALTER TABLE question_jobs ADD COLUMN request_fingerprint TEXT"
                )
            migration_timestamp = _utc_timestamp()
            self._conn.execute(
                "UPDATE question_jobs SET created_at = ? "
                "WHERE created_at IS NULL OR created_at = ''",
                (migration_timestamp,),
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_question_jobs_idempotency_key_hash "
                "ON question_jobs (idempotency_key_hash)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_question_jobs_created_at "
                "ON question_jobs (created_at)"
            )
            self._conn.commit()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def create_job(self, request_id: str) -> QuestionJob:
        """Create and store a pending job."""
        job = QuestionJob(request_id=request_id, status="pending")
        return self._insert_job(job)

    def create_completed_job(
        self,
        request_id: str,
        result: QuestionResult,
        *,
        idempotency_key_hash: str | None = None,
        request_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a completed job without persisting an intermediate state."""
        if result.request_id != request_id:
            raise ValueError(
                f"Result request ID does not match job: {result.request_id} != {request_id}"
            )
        job = QuestionJob(request_id=request_id, status="completed", result=result)
        return self._insert_job(
            job,
            idempotency_key_hash=idempotency_key_hash,
            request_fingerprint=request_fingerprint,
        )

    def create_failed_job(
        self,
        request_id: str,
        error_category: Literal["processing_error"],
        *,
        idempotency_key_hash: str | None = None,
        request_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a failed job without persisting an intermediate state."""
        job = QuestionJob(
            request_id=request_id,
            status="failed",
            error_category=error_category,
        )
        return self._insert_job(
            job,
            idempotency_key_hash=idempotency_key_hash,
            request_fingerprint=request_fingerprint,
        )

    def _insert_job(
        self,
        job: QuestionJob,
        *,
        idempotency_key_hash: str | None = None,
        request_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a new job record and preserve duplicate-ID errors."""
        if (idempotency_key_hash is None) != (request_fingerprint is None):
            raise ValueError("Idempotency key and request fingerprint must be provided together")

        with self._lock:
            try:
                values = (
                    job.request_id,
                    job.model_dump_json(),
                    _utc_timestamp(),
                    idempotency_key_hash,
                    request_fingerprint,
                )
                insert = (
                    "INSERT INTO question_jobs "
                    "(request_id, job_json, created_at, idempotency_key_hash, "
                    "request_fingerprint) VALUES (?, ?, ?, ?, ?)"
                )
                if idempotency_key_hash is None:
                    self._conn.execute(insert, values)
                else:
                    cursor = self._conn.execute(
                        insert + " ON CONFLICT(idempotency_key_hash) DO NOTHING",
                        values,
                    )
                    if cursor.rowcount == 0:
                        row = self._conn.execute(
                            "SELECT job_json, request_fingerprint FROM question_jobs "
                            "WHERE idempotency_key_hash = ?",
                            (idempotency_key_hash,),
                        ).fetchone()
                        self._conn.commit()
                        if row is None:
                            raise sqlite3.DatabaseError(
                                "Unable to load the existing idempotency record"
                            )
                        if row[1] != request_fingerprint:
                            raise IdempotencyKeyReuseError() from None
                        return QuestionJob.model_validate_json(row[0])
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Job already exists: {job.request_id}") from exc
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

    def get_by_idempotency_key(
        self, idempotency_key_hash: str
    ) -> tuple[QuestionJob, str] | None:
        """Load a terminal job and its request fingerprint by hashed key."""
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json, request_fingerprint FROM question_jobs "
                "WHERE idempotency_key_hash = ?",
                (idempotency_key_hash,),
            ).fetchone()
        if row is None:
            return None
        if row[1] is None:
            raise sqlite3.DatabaseError("Stored idempotency record is incomplete")
        return QuestionJob.model_validate_json(row[0]), row[1]

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
        """Return whether the question-job table can be queried."""
        try:
            with self._lock:
                self._conn.execute(
                    "SELECT request_id FROM question_jobs LIMIT 1"
                ).fetchone()
            return True
        except sqlite3.Error:
            return False

    def verify_integrity(self) -> None:
        """Run SQLite's quick structural check and fail on detected corruption."""
        with self._lock:
            rows = self._conn.execute("PRAGMA quick_check").fetchall()
        if rows != [("ok",)]:
            raise sqlite3.DatabaseError("SQLite quick_check failed")

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()


def _utc_timestamp() -> str:
    """Return a sortable UTC timestamp with fixed precision."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
