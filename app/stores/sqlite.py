"""SQLite implementation of question job persistence."""

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Literal

from app.domain.errors import (
    IdempotencyKeyReuseError,
    IncompleteIdempotencyRecordError,
)
from app.domain.questions import QuestionJob, QuestionResult

# Both idempotency columns must be null together, or both set together.
_IDEMPOTENCY_PAIR_CHECK = (
    "("
    "(idempotency_key_hash IS NULL AND question_fingerprint IS NULL) OR "
    "(idempotency_key_hash IS NOT NULL AND question_fingerprint IS NOT NULL)"
    ")"
)
_QUESTION_JOBS_TABLE_SQL = f"""
                CREATE TABLE question_jobs (
                    question_id TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    idempotency_key_hash TEXT,
                    question_fingerprint TEXT,
                    CHECK {_IDEMPOTENCY_PAIR_CHECK}
                )
                """


class SQLiteQuestionJobStore:
    """Persist question jobs as JSON records in SQLite."""

    def __init__(self, db_path: str = ":memory:", timeout: float = 5.0) -> None:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
        self._lock = threading.Lock()
        try:
            self._ensure_table()
        except BaseException:
            self._conn.close()
            raise

    def _ensure_table(self) -> None:
        """Create the jobs table and enable WAL for file-backed databases."""
        with self._lock:
            self._conn.execute(
                _QUESTION_JOBS_TABLE_SQL.replace(
                    "CREATE TABLE question_jobs",
                    "CREATE TABLE IF NOT EXISTS question_jobs",
                )
            )
            migrate_question_job_names(self._conn, "question_jobs")
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
            if "question_fingerprint" not in columns:
                self._conn.execute(
                    "ALTER TABLE question_jobs ADD COLUMN question_fingerprint TEXT"
                )
            migration_timestamp = _utc_timestamp()
            self._conn.execute(
                "UPDATE question_jobs SET created_at = ? "
                "WHERE created_at IS NULL OR created_at = ''",
                (migration_timestamp,),
            )
            _repair_and_ensure_idempotency_pair_check(self._conn)
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

    def create_job(self, question_id: str) -> QuestionJob:
        """Create and store a pending job."""
        job = QuestionJob(question_id=question_id, status="pending")
        return self._insert_job(job)

    def create_completed_job(
        self,
        question_id: str,
        result: QuestionResult,
        *,
        idempotency_key_hash: str | None = None,
        question_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a completed job without persisting an intermediate state."""
        if result.question_id != question_id:
            raise ValueError(
                f"Result question ID does not match job: {result.question_id} != {question_id}"
            )
        job = QuestionJob(question_id=question_id, status="completed", result=result)
        return self._insert_job(
            job,
            idempotency_key_hash=idempotency_key_hash,
            question_fingerprint=question_fingerprint,
        )

    def create_failed_job(
        self,
        question_id: str,
        error_category: Literal["processing_error"],
        *,
        idempotency_key_hash: str | None = None,
        question_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a failed job without persisting an intermediate state."""
        job = QuestionJob(
            question_id=question_id,
            status="failed",
            error_category=error_category,
        )
        return self._insert_job(
            job,
            idempotency_key_hash=idempotency_key_hash,
            question_fingerprint=question_fingerprint,
        )

    def _insert_job(
        self,
        job: QuestionJob,
        *,
        idempotency_key_hash: str | None = None,
        question_fingerprint: str | None = None,
    ) -> QuestionJob:
        """Insert a new job record and preserve duplicate-ID errors."""
        if (idempotency_key_hash is None) != (question_fingerprint is None):
            raise ValueError("Idempotency key and question fingerprint must be provided together")

        with self._lock:
            try:
                values = (
                    job.question_id,
                    job.model_dump_json(),
                    _utc_timestamp(),
                    idempotency_key_hash,
                    question_fingerprint,
                )
                insert = (
                    "INSERT INTO question_jobs "
                    "(question_id, job_json, created_at, idempotency_key_hash, "
                    "question_fingerprint) VALUES (?, ?, ?, ?, ?)"
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
                            "SELECT job_json, question_fingerprint FROM question_jobs "
                            "WHERE idempotency_key_hash = ?",
                            (idempotency_key_hash,),
                        ).fetchone()
                        self._conn.commit()
                        if row is None or row[1] is None:
                            raise IncompleteIdempotencyRecordError() from None
                        if row[1] != question_fingerprint:
                            raise IdempotencyKeyReuseError() from None
                        return QuestionJob.model_validate_json(rename_stored_question_id(row[0]))
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Job already exists: {job.question_id}") from exc
        return job

    def get_job(self, question_id: str) -> QuestionJob | None:
        """Load a job, returning None when it does not exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json FROM question_jobs WHERE question_id = ?", (question_id,)
            ).fetchone()
        if row is None:
            return None
        return QuestionJob.model_validate_json(rename_stored_question_id(row[0]))

    def get_by_idempotency_key(
        self, idempotency_key_hash: str
    ) -> tuple[QuestionJob, str] | None:
        """Load a terminal job and its question fingerprint by hashed key."""
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json, question_fingerprint FROM question_jobs "
                "WHERE idempotency_key_hash = ?",
                (idempotency_key_hash,),
            ).fetchone()
        if row is None:
            return None
        if row[1] is None:
            raise IncompleteIdempotencyRecordError() from None
        return QuestionJob.model_validate_json(rename_stored_question_id(row[0])), row[1]

    def complete_job(self, question_id: str, result: QuestionResult) -> QuestionJob:
        """Persist a completed result for an existing job."""
        if result.question_id != question_id:
            raise ValueError(
                f"Result question ID does not match job: {result.question_id} != {question_id}"
            )
        job = QuestionJob(question_id=question_id, status="completed", result=result)
        self._save_job(job)
        return job

    def fail_job(
        self, question_id: str, error_category: Literal["processing_error"]
    ) -> QuestionJob:
        """Persist a safe failure category for an existing job."""
        job = QuestionJob(
            question_id=question_id,
            status="failed",
            error_category=error_category,
        )
        self._save_job(job)
        return job

    def _save_job(self, job: QuestionJob) -> None:
        """Replace an existing job record."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE question_jobs SET job_json = ? WHERE question_id = ?",
                (job.model_dump_json(), job.question_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Job not found: {job.question_id}")
            self._conn.commit()

    def is_ready(self) -> bool:
        """Return whether the question-job table can be queried."""
        try:
            with self._lock:
                self._conn.execute(
                    "SELECT question_id FROM question_jobs LIMIT 1"
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


def _idempotency_pair_check_present(conn: sqlite3.Connection) -> bool:
    """Return whether ``question_jobs`` already has the paired-column CHECK."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'question_jobs'"
    ).fetchone()
    if row is None or row[0] is None:
        return False
    normalized = " ".join(row[0].split())
    return _IDEMPOTENCY_PAIR_CHECK in normalized


def _repair_and_ensure_idempotency_pair_check(conn: sqlite3.Connection) -> None:
    """Clear inconsistent idempotency columns and rebuild with a CHECK constraint."""
    conn.execute(
        "UPDATE question_jobs "
        "SET idempotency_key_hash = NULL, question_fingerprint = NULL "
        "WHERE (idempotency_key_hash IS NULL AND question_fingerprint IS NOT NULL) "
        "OR (idempotency_key_hash IS NOT NULL AND question_fingerprint IS NULL)"
    )
    if _idempotency_pair_check_present(conn):
        return
    conn.execute("ALTER TABLE question_jobs RENAME TO question_jobs_legacy_idempotency")
    conn.execute(_QUESTION_JOBS_TABLE_SQL)
    conn.execute(
        "INSERT INTO question_jobs "
        "(question_id, job_json, created_at, idempotency_key_hash, question_fingerprint) "
        "SELECT question_id, job_json, created_at, idempotency_key_hash, question_fingerprint "
        "FROM question_jobs_legacy_idempotency"
    )
    conn.execute("DROP TABLE question_jobs_legacy_idempotency")


def migrate_question_job_names(conn: sqlite3.Connection, table: str) -> None:
    """Rename stored job IDs from ``request_id`` to ``question_id``.

    Older databases used ``request_id`` for the question job and
    ``request_fingerprint`` for the question-text hash. Those names now belong
    only to the HTTP correlation ID, so existing rows are rewritten in place.
    """
    if table not in {"question_jobs", "archived_question_jobs"}:
        raise ValueError(f"Unknown job table: {table}")
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if not columns:
        return
    if "request_id" in columns and "question_id" not in columns:
        conn.execute(f"ALTER TABLE {table} RENAME COLUMN request_id TO question_id")
    if "request_fingerprint" in columns and "question_fingerprint" not in columns:
        conn.execute(
            f"ALTER TABLE {table} RENAME COLUMN request_fingerprint TO question_fingerprint"
        )
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "question_id" not in columns or "job_json" not in columns:
        return
    rows = conn.execute(f"SELECT question_id, job_json FROM {table}").fetchall()
    for question_id, job_json in rows:
        updated = rename_stored_question_id(job_json)
        if updated != job_json:
            conn.execute(
                f"UPDATE {table} SET job_json = ? WHERE question_id = ?",
                (updated, question_id),
            )


def rename_stored_question_id(job_json: str) -> str:
    """Return job JSON whose job and result IDs use ``question_id``."""
    try:
        payload = json.loads(job_json)
    except json.JSONDecodeError:
        return job_json
    if not isinstance(payload, dict):
        return job_json
    changed = _rename_request_id_key(payload)
    result = payload.get("result")
    if isinstance(result, dict):
        changed = _rename_request_id_key(result) or changed
    if not changed:
        return job_json
    return json.dumps(payload, separators=(",", ":"))


def _rename_request_id_key(payload: dict) -> bool:
    if "request_id" not in payload or "question_id" in payload:
        return False
    renamed = {
        ("question_id" if key == "request_id" else key): value for key, value in payload.items()
    }
    payload.clear()
    payload.update(renamed)
    return True


def _utc_timestamp() -> str:
    """Return a sortable UTC timestamp with fixed precision."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
