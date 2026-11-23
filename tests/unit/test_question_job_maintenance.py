"""Unit tests for SQLite question-job backup, archive, and restore maintenance."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.domain.questions import (
    ModelMetrics,
    QuestionJob,
    QuestionResult,
    RetrievalMetrics,
)
from app.stores.sqlite import SQLiteQuestionJobStore
from scripts import question_job_maintenance as maintenance
from scripts.question_job_maintenance import (
    MaintenanceError,
    _check_database,
    _rotate_snapshots,
    _write_archive_rows,
    archive_old_jobs,
    backup_archives,
    backup_live_database,
    main,
    verify_restore,
)


def _result(question_id: str, answer: str = "A test answer") -> QuestionResult:
    """Build a minimal completed result for maintenance fixtures.

    :param question_id: Identifier assigned to the synthetic result.
    :param answer: Answer text stored on the result.
    :returns: A ``QuestionResult`` with empty citations and zero retrieval counts."""
    return QuestionResult(
        question_id=question_id,
        answer=answer,
        citations=[],
        retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
        model=ModelMetrics(
            provider="test", name="test", input_tokens=0, output_tokens=0
        ),
    )


# Fixed "now" for retention math: live cutoff = now-7d, archive cutoff = now-37d.
_FIXED_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
# Older than live retention, newer than live+archive retention (survives prune).
_ARCHIVABLE_CREATED_AT = "2026-06-01T00:00:00.000000Z"
# Older than live+archive retention (eligible for prune).
_EXPIRED_CREATED_AT = "2025-01-01T00:00:00.000000Z"


@pytest.fixture
def maintenance_paths(tmp_path, monkeypatch):
    """Point maintenance settings at isolated temporary directories.

    Uses short retention windows and a frozen clock so age-based archive and
    prune cases stay deterministic.

    :param tmp_path: Pytest temporary directory for this test.
    :param monkeypatch: Fixture used to override module-level settings.
    :returns: Mapping of ``db``, ``backup``, and ``archive`` paths."""
    db_path = tmp_path / "questions.db"
    backup_dir = tmp_path / "backups"
    archive_dir = tmp_path / "archives"
    monkeypatch.setattr(maintenance.settings, "questions_db_path", str(db_path))
    monkeypatch.setattr(maintenance.settings, "questions_backup_dir", str(backup_dir))
    monkeypatch.setattr(maintenance.settings, "questions_archive_dir", str(archive_dir))
    monkeypatch.setattr(maintenance.settings, "questions_live_retention_days", 7)
    monkeypatch.setattr(maintenance.settings, "questions_archive_retention_days", 30)
    monkeypatch.setattr(maintenance, "_utc_now", lambda: _FIXED_NOW)
    return {"db": db_path, "backup": backup_dir, "archive": archive_dir}


def _seed_live_db(db_path: Path) -> SQLiteQuestionJobStore:
    """Create a live job store at ``db_path``.

    :param db_path: Filesystem path for the SQLite database.
    :returns: An open store the caller must close."""
    return SQLiteQuestionJobStore(str(db_path))


def _backdate(db_path: Path, question_id: str, created_at: str) -> None:
    """Overwrite ``created_at`` for a live job so retention cutoffs can select it.

    :param db_path: Live database path.
    :param question_id: Job identifier to update.
    :param created_at: Timestamp string in the store's UTC ``Z`` format."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE question_jobs SET created_at = ? WHERE question_id = ?",
            (created_at, question_id),
        )
        conn.commit()


def _live_count(db_path: Path) -> int:
    """Return the number of rows in the live ``question_jobs`` table.

    :param db_path: Live database path.
    :returns: Row count."""
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM question_jobs").fetchone()[0]


def _live_ids(db_path: Path) -> set[str]:
    """Return live question IDs.

    :param db_path: Live database path.
    :returns: Set of ``question_id`` values."""
    with sqlite3.connect(db_path) as conn:
        return {row[0] for row in conn.execute("SELECT question_id FROM question_jobs")}


def _archive_ids(archive_dir: Path) -> set[str]:
    """Return question IDs stored across monthly archive databases.

    :param archive_dir: Root archive directory.
    :returns: Set of archived ``question_id`` values."""
    ids: set[str] = set()
    for path in archive_dir.glob("jobs-*.sqlite3"):
        with sqlite3.connect(path) as conn:
            ids.update(
                row[0]
                for row in conn.execute(
                    "SELECT question_id FROM archived_question_jobs"
                )
            )
    return ids


def _file_mode(path: Path) -> int:
    """Return the permission bits for ``path``.

    :param path: File whose mode should be inspected.
    :returns: Mode bits masked to ``0o777``."""
    return path.stat().st_mode & 0o777


# --- Happy paths ---


def test_backup_live_database_creates_verified_snapshot(maintenance_paths):
    """H1: Snapshot the live database with matching row count and restricted mode."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("job-1", _result("job-1"))
    finally:
        store.close()

    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    snapshot = backup_live_database(now=now)

    assert snapshot == maintenance_paths["backup"] / "live" / (
        now.strftime(maintenance.SNAPSHOT_FORMAT) + ".sqlite3"
    )
    assert snapshot.is_file()
    assert _file_mode(snapshot) == 0o600
    with sqlite3.connect(snapshot) as conn:
        assert conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]
        assert conn.execute("SELECT COUNT(*) FROM question_jobs").fetchone()[0] == 1


def test_verify_restore_accepts_explicit_snapshot(maintenance_paths):
    """H2: Restore-check an explicit live snapshot and return its job count."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("job-1", _result("job-1"))
        store.create_completed_job("job-2", _result("job-2"))
    finally:
        store.close()
    snapshot = backup_live_database(now=datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))

    assert verify_restore(snapshot) == 2


def test_verify_restore_defaults_to_newest_live_snapshot(maintenance_paths):
    """H3: Omitting the snapshot path uses the newest live backup."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("job-1", _result("job-1"))
    finally:
        store.close()
    older = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    newer = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    backup_live_database(now=older)
    backup_live_database(now=newer)

    assert verify_restore(None) == 1


def test_backup_archives_snapshots_monthly_archive_database(maintenance_paths):
    """H4: Snapshot each monthly archive database once."""
    archive_path = maintenance_paths["archive"] / "jobs-2026-01.sqlite3"
    _write_archive_rows(
        archive_path,
        [
            (
                "archived-1",
                QuestionJob(
                    question_id="archived-1",
                    status="completed",
                    result=_result("archived-1"),
                ).model_dump_json(),
                "2026-01-15T00:00:00.000000Z",
            )
        ],
    )

    count = backup_archives(now=datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))

    assert count == 1
    snapshots = list(
        (maintenance_paths["backup"] / "archives" / "jobs-2026-01").glob("*.sqlite3")
    )
    assert len(snapshots) == 1
    assert _file_mode(snapshots[0]) == 0o600


def test_backup_archives_returns_zero_when_archive_dir_missing(maintenance_paths):
    """H5: Missing archive directory is a no-op success."""
    assert not maintenance_paths["archive"].exists()

    assert backup_archives(now=datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)) == 0


def test_backup_archives_suppresses_duplicate_digest_within_retention(maintenance_paths):
    """H6: An unchanged archive within the monthly window does not add a snapshot."""
    archive_path = maintenance_paths["archive"] / "jobs-2026-01.sqlite3"
    _write_archive_rows(
        archive_path,
        [
            (
                "archived-1",
                QuestionJob(
                    question_id="archived-1",
                    status="failed",
                    error_category="processing_error",
                ).model_dump_json(),
                "2026-01-15T00:00:00.000000Z",
            )
        ],
    )
    first = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    second = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

    assert backup_archives(now=first) == 1
    assert backup_archives(now=second) == 0
    snapshots = list(
        (maintenance_paths["backup"] / "archives" / "jobs-2026-01").glob("*.sqlite3")
    )
    assert len(snapshots) == 1


def test_archive_dry_run_reports_nonterminal_without_changes(
    maintenance_paths, capsys
):
    """H7: Dry-run counts eligible work and leaves live and archive data untouched."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("old-done", _result("old-done"))
        store.create_job("old-pending")
    finally:
        store.close()
    _backdate(maintenance_paths["db"], "old-done", _ARCHIVABLE_CREATED_AT)
    _backdate(maintenance_paths["db"], "old-pending", _ARCHIVABLE_CREATED_AT)
    before_ids = _live_ids(maintenance_paths["db"])

    archived, pruned, nonterminal = archive_old_jobs(dry_run=True)

    assert (archived, pruned, nonterminal) == (0, 0, 1)
    assert _live_ids(maintenance_paths["db"]) == before_ids
    assert list(maintenance_paths["archive"].glob("jobs-*.sqlite3")) == []
    captured = capsys.readouterr().out
    assert "Would archive 1 terminal jobs" in captured
    assert "leave 1 nonterminal jobs" in captured


def test_archive_moves_old_terminal_jobs_and_keeps_young_and_pending(maintenance_paths):
    """H8/H9: Archive old completed/failed jobs; keep young and pending live rows."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("old-done", _result("old-done", answer="done"))
        store.create_failed_job("old-failed", "processing_error")
        store.create_job("old-pending")
        store.create_completed_job("young-done", _result("young-done"))
    finally:
        store.close()
    for question_id in ("old-done", "old-failed", "old-pending"):
        _backdate(maintenance_paths["db"], question_id, _ARCHIVABLE_CREATED_AT)

    archived, pruned, nonterminal = archive_old_jobs()

    assert archived == 2
    assert pruned == 0
    assert nonterminal == 1
    assert _live_ids(maintenance_paths["db"]) == {"old-pending", "young-done"}
    assert _archive_ids(maintenance_paths["archive"]) == {"old-done", "old-failed"}
    assert (maintenance_paths["backup"] / "live").exists()


def test_archive_is_idempotent_for_already_archived_rows(maintenance_paths):
    """H10: Re-archiving matching terminal rows does not raise or duplicate archives."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("old-done", _result("old-done"))
    finally:
        store.close()
    _backdate(maintenance_paths["db"], "old-done", _ARCHIVABLE_CREATED_AT)

    first = archive_old_jobs()
    # Re-insert the same terminal job and backdate it again.
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("old-done", _result("old-done"))
    finally:
        store.close()
    _backdate(maintenance_paths["db"], "old-done", _ARCHIVABLE_CREATED_AT)
    second = archive_old_jobs()

    assert first[0] == 1
    assert second[0] == 1
    with sqlite3.connect(
        next(maintenance_paths["archive"].glob("jobs-*.sqlite3"))
    ) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM archived_question_jobs WHERE question_id = ?",
            ("old-done",),
        ).fetchone()[0]
    assert count == 1


def test_archive_prunes_expired_archive_rows_and_removes_empty_files(maintenance_paths):
    """H11: Rows older than live+archive retention are pruned from archives."""
    store = _seed_live_db(maintenance_paths["db"])
    store.close()
    archive_path = maintenance_paths["archive"] / "jobs-2025-01.sqlite3"
    _write_archive_rows(
        archive_path,
        [
            (
                "expired-1",
                QuestionJob(
                    question_id="expired-1",
                    status="completed",
                    result=_result("expired-1"),
                ).model_dump_json(),
                _EXPIRED_CREATED_AT,
            )
        ],
    )
    assert archive_path.is_file()

    archived, pruned, nonterminal = archive_old_jobs()

    assert archived == 0
    assert pruned == 1
    assert nonterminal == 0
    assert not archive_path.exists()
    assert _archive_ids(maintenance_paths["archive"]) == set()


def test_main_backup_live_and_archive_dry_run_exit_zero(maintenance_paths, capsys):
    """H12: CLI backup-live and archive --dry-run return process exit code 0."""
    store = _seed_live_db(maintenance_paths["db"])
    try:
        store.create_completed_job("job-1", _result("job-1"))
    finally:
        store.close()

    assert main(["backup-live"]) == 0
    assert "Created live database snapshot:" in capsys.readouterr().out
    assert main(["archive", "--dry-run"]) == 0


# --- Negative paths ---


def test_backup_live_missing_database_raises(maintenance_paths):
    """N1: Live backup fails closed when the database file is absent."""
    assert not maintenance_paths["db"].exists()

    with pytest.raises(MaintenanceError, match="does not exist"):
        backup_live_database(now=datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))


def test_main_backup_live_missing_database_exits_one(maintenance_paths, capsys):
    """N1b: CLI maps missing-database failures to exit code 1 without a traceback."""
    assert main(["backup-live"]) == 1
    err = capsys.readouterr().err
    assert "MaintenanceError" in err
    assert "Traceback" not in err


def test_verify_restore_without_snapshots_raises(maintenance_paths):
    """N2: Restore-check requires at least one live snapshot."""
    maintenance_paths["backup"].mkdir(parents=True)
    (maintenance_paths["backup"] / "live").mkdir()

    with pytest.raises(MaintenanceError, match="No live database snapshots"):
        verify_restore(None)


def test_archive_dry_run_missing_database_raises(maintenance_paths):
    """N3: Dry-run fails when the live database file does not exist."""
    with pytest.raises(MaintenanceError, match="does not exist"):
        archive_old_jobs(dry_run=True)


def test_archive_dry_run_rejects_unmigrated_schema(maintenance_paths):
    """N4: Dry-run requires the migrated live schema including created_at."""
    maintenance_paths["db"].parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(maintenance_paths["db"]) as conn:
        conn.execute(
            "CREATE TABLE question_jobs ("
            "question_id TEXT PRIMARY KEY, job_json TEXT NOT NULL)"
        )
        conn.commit()

    with pytest.raises(MaintenanceError, match="migrate the database"):
        archive_old_jobs(dry_run=True)


def test_archive_rejects_undecodable_job_json(maintenance_paths):
    """N5: Undecodable live job JSON fails before rows are moved."""
    store = _seed_live_db(maintenance_paths["db"])
    store.close()
    with sqlite3.connect(maintenance_paths["db"]) as conn:
        conn.execute(
            "INSERT INTO question_jobs "
            "(question_id, job_json, created_at, idempotency_key_hash, "
            "question_fingerprint) VALUES (?, ?, ?, ?, ?)",
            ("bad-1", "{not-json", _ARCHIVABLE_CREATED_AT, None, None),
        )
        conn.commit()

    with pytest.raises(MaintenanceError, match="could not be decoded"):
        archive_old_jobs()
    assert _live_ids(maintenance_paths["db"]) == {"bad-1"}
    assert _archive_ids(maintenance_paths["archive"]) == set()


def test_write_archive_rows_rejects_mismatched_existing_payload(maintenance_paths):
    """N6: An existing archived row with different payload fails closed."""
    archive_path = maintenance_paths["archive"] / "jobs-2026-01.sqlite3"
    original = QuestionJob(
        question_id="job-1",
        status="completed",
        result=_result("job-1", answer="original"),
    ).model_dump_json()
    conflict = QuestionJob(
        question_id="job-1",
        status="completed",
        result=_result("job-1", answer="different"),
    ).model_dump_json()
    created_at = "2026-01-15T00:00:00.000000Z"
    _write_archive_rows(archive_path, [("job-1", original, created_at)])

    with pytest.raises(MaintenanceError, match="does not match its source"):
        _write_archive_rows(archive_path, [("job-1", conflict, created_at)])


def test_check_database_raises_when_quick_check_fails():
    """N7: Integrity helper raises a safe MaintenanceError on failed quick_check."""

    class FakeCursor:
        def fetchall(self):
            return [("not ok",)]

    class FakeConn:
        def execute(self, sql):
            assert sql == "PRAGMA quick_check"
            return FakeCursor()

    with pytest.raises(MaintenanceError, match="quick_check failed"):
        _check_database(FakeConn())


def test_rotate_snapshots_keeps_hourly_and_removes_expired(maintenance_paths):
    """N8: Rotation keeps recent hourly snapshots and drops aged-out files."""
    directory = maintenance_paths["backup"] / "live"
    directory.mkdir(parents=True)
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    keep_recent = now - timedelta(hours=1)
    keep_daily = now - timedelta(days=2)
    drop_old = now - timedelta(days=400)
    paths = {}
    for stamp in (keep_recent, keep_daily, drop_old):
        path = directory / f"{stamp.strftime(maintenance.SNAPSHOT_FORMAT)}.sqlite3"
        path.write_bytes(b"snapshot")
        paths[stamp] = path

    removed = _rotate_snapshots(directory, now)

    assert removed == 1
    assert paths[keep_recent].exists()
    assert paths[keep_daily].exists()
    assert not paths[drop_old].exists()


def test_main_restore_check_missing_snapshot_exits_one(maintenance_paths, capsys):
    """N9: CLI restore-check returns 1 for a missing snapshot path."""
    missing = maintenance_paths["backup"] / "missing.sqlite3"

    assert main(["restore-check", str(missing)]) == 1
    err = capsys.readouterr().err
    assert "MaintenanceError" in err
    assert "Traceback" not in err


def test_main_without_command_exits_via_argparse():
    """N10: Missing required subcommand is rejected by argparse."""
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == 2
