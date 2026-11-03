"""Back up, archive, and expire SQLite question-job records.

Run ``python -m scripts.question_job_maintenance --help`` for commands.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from app.core.config import settings
from app.models.questions import QuestionJob
from app.stores.sqlite_question_jobs import SQLiteQuestionJobStore

BATCH_SIZE = 500
SNAPSHOT_FORMAT = "%Y%m%dT%H%M%S%fZ"
SNAPSHOT_PATTERN = re.compile(r"\d{8}T\d{12}Z\Z")
HOURLY_RETENTION = timedelta(hours=24)
DAILY_RETENTION = timedelta(days=30)
WEEKLY_RETENTION = timedelta(weeks=12)
MONTHLY_RETENTION = timedelta(days=365)


class MaintenanceError(RuntimeError):
    """A safe operational error that does not include stored job contents."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def _check_database(conn: sqlite3.Connection) -> None:
    if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise MaintenanceError("SQLite quick_check failed")


def _open_readonly_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise MaintenanceError("Question database does not exist")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def _snapshot_database(source_path: Path, destination: Path) -> None:
    """Write a consistent, verified SQLite snapshot and atomically publish it."""
    if not source_path.is_file():
        raise MaintenanceError("Source database does not exist")
    _ensure_private_directory(destination.parent)

    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    source_conn: sqlite3.Connection | None = None
    destination_conn: sqlite3.Connection | None = None
    try:
        source_uri = f"{source_path.resolve().as_uri()}?mode=ro"
        source_conn = sqlite3.connect(source_uri, uri=True, timeout=30.0)
        destination_conn = sqlite3.connect(temporary_path, timeout=30.0)
        source_conn.backup(destination_conn, pages=256, sleep=0.1)
        _check_database(destination_conn)
        destination_conn.close()
        destination_conn = None
        source_conn.close()
        source_conn = None
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    finally:
        if destination_conn is not None:
            destination_conn.close()
        if source_conn is not None:
            source_conn.close()
        temporary_path.unlink(missing_ok=True)
        Path(f"{temporary_path}-wal").unlink(missing_ok=True)
        Path(f"{temporary_path}-shm").unlink(missing_ok=True)


def _snapshot_time(path: Path) -> datetime | None:
    if not SNAPSHOT_PATTERN.fullmatch(path.stem):
        return None
    return datetime.strptime(path.stem, SNAPSHOT_FORMAT).replace(tzinfo=UTC)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rotate_snapshots(directory: Path, now: datetime) -> int:
    """Keep hourly, daily, weekly, and monthly snapshots in their age bands."""
    snapshots = [
        (path, _snapshot_time(path))
        for path in directory.glob("*.sqlite3")
    ]
    snapshots = [(path, timestamp) for path, timestamp in snapshots if timestamp]
    keep: set[Path] = set()
    daily: dict[tuple[int, int, int], tuple[datetime, Path]] = {}
    weekly: dict[tuple[int, int], tuple[datetime, Path]] = {}
    monthly: dict[tuple[int, int], tuple[datetime, Path]] = {}

    for path, timestamp in snapshots:
        age = now - timestamp
        if age <= HOURLY_RETENTION:
            keep.add(path)
        elif age <= DAILY_RETENTION:
            key = (timestamp.year, timestamp.month, timestamp.day)
            if key not in daily or timestamp > daily[key][0]:
                daily[key] = (timestamp, path)
        elif age <= WEEKLY_RETENTION:
            week = timestamp.isocalendar()
            key = (week.year, week.week)
            if key not in weekly or timestamp > weekly[key][0]:
                weekly[key] = (timestamp, path)
        elif age <= MONTHLY_RETENTION:
            key = (timestamp.year, timestamp.month)
            if key not in monthly or timestamp > monthly[key][0]:
                monthly[key] = (timestamp, path)

    keep.update(path for _, path in daily.values())
    keep.update(path for _, path in weekly.values())
    keep.update(path for _, path in monthly.values())

    removed = 0
    for path, timestamp in snapshots:
        if path not in keep:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def backup_live_database(now: datetime | None = None) -> Path:
    """Create a consistent live-database snapshot and apply GFS retention."""
    source = Path(settings.questions_db_path)
    now = now or _utc_now()
    directory = Path(settings.questions_backup_dir) / "live"
    stamp = now.astimezone(UTC).strftime(SNAPSHOT_FORMAT)
    destination = directory / f"{stamp}.sqlite3"
    _snapshot_database(source, destination)
    _rotate_snapshots(directory, now.astimezone(UTC))
    return destination


def backup_archives(now: datetime | None = None) -> int:
    """Snapshot monthly archive databases and rotate their backup tiers."""
    archive_dir = Path(settings.questions_archive_dir)
    if not archive_dir.exists():
        return 0
    now = now or _utc_now()
    count = 0
    for source in sorted(archive_dir.glob("jobs-*.sqlite3")):
        directory = Path(settings.questions_backup_dir) / "archives" / source.stem
        stamp = now.astimezone(UTC).strftime(SNAPSHOT_FORMAT)
        destination = directory / f"{stamp}.sqlite3"
        _snapshot_database(source, destination)
        previous = [
            path
            for path in directory.glob("*.sqlite3")
            if path != destination and _snapshot_time(path) is not None
        ]
        if previous:
            latest = max(previous, key=lambda path: path.stem)
            latest_time = _snapshot_time(latest)
            if (
                latest_time is not None
                and now.astimezone(UTC) - latest_time <= MONTHLY_RETENTION
                and _file_digest(destination) == _file_digest(latest)
            ):
                destination.unlink(missing_ok=True)
            else:
                count += 1
        else:
            count += 1
        _rotate_snapshots(directory, now.astimezone(UTC))
    return count


def verify_restore(snapshot: Path | None = None) -> int:
    """Restore a backup into a temporary database and verify its schema and rows."""
    if snapshot is None:
        live_backup_dir = Path(settings.questions_backup_dir) / "live"
        candidates = [
            path
            for path in live_backup_dir.glob("*.sqlite3")
            if _snapshot_time(path) is not None
        ]
        if not candidates:
            raise MaintenanceError("No live database snapshots are available")
        snapshot = max(candidates, key=lambda path: path.stem)

    with tempfile.TemporaryDirectory(prefix="question-job-restore-") as temp_dir:
        restored_path = Path(temp_dir) / "restored.sqlite3"
        _snapshot_database(snapshot, restored_path)
        store = SQLiteQuestionJobStore(str(restored_path), timeout=30.0)
        try:
            store.verify_integrity()
            if not store.is_ready():
                raise MaintenanceError("Restored question-job table is unavailable")
        finally:
            store.close()
        with sqlite3.connect(restored_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM question_jobs").fetchone()[0]
    return count


def _archive_path(archive_dir: Path, created_at: str) -> Path:
    try:
        month = _parse_timestamp(created_at).strftime("%Y-%m")
    except ValueError as exc:
        raise MaintenanceError("A job has an invalid creation timestamp") from exc
    return archive_dir / f"jobs-{month}.sqlite3"


def _write_archive_rows(
    archive_path: Path,
    rows: Sequence[tuple[str, str, str]],
) -> None:
    """Copy terminal jobs idempotently and verify archived rows before deletion."""
    _ensure_private_directory(archive_path.parent)
    conn = sqlite3.connect(archive_path, timeout=30.0)
    try:
        _check_database(conn)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS archived_question_jobs ("
            "request_id TEXT PRIMARY KEY, job_json TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_question_jobs_created_at "
            "ON archived_question_jobs (created_at)"
        )
        with conn:
            for request_id, job_json, created_at in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO archived_question_jobs "
                    "(request_id, job_json, created_at) VALUES (?, ?, ?)",
                    (request_id, job_json, created_at),
                )
                archived = conn.execute(
                    "SELECT job_json, created_at FROM archived_question_jobs "
                    "WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if archived != (job_json, created_at):
                    raise MaintenanceError("An archived job does not match its source")
        _check_database(conn)
    finally:
        conn.close()
    os.chmod(archive_path, 0o600)


def _archive_batch(
    live_conn: sqlite3.Connection,
    archive_dir: Path,
    rows: Sequence[tuple[str, str, str]],
) -> tuple[int, int]:
    groups: dict[Path, list[tuple[str, str, str]]] = defaultdict(list)
    nonterminal = 0
    for row in rows:
        try:
            job = QuestionJob.model_validate_json(row[1])
        except Exception as exc:
            raise MaintenanceError("A job record could not be decoded") from exc
        if job.status in {"completed", "failed"}:
            groups[_archive_path(archive_dir, row[2])].append(row)
        else:
            nonterminal += 1

    archived_count = 0
    for archive_path, group in groups.items():
        _write_archive_rows(archive_path, group)
        before = live_conn.total_changes
        with live_conn:
            live_conn.executemany(
                "DELETE FROM question_jobs "
                "WHERE request_id = ? AND job_json = ? AND created_at = ?",
                group,
            )
        deleted = live_conn.total_changes - before
        if deleted != len(group):
            raise MaintenanceError(
                "Live jobs changed during archival; no further rows moved"
            )
        archived_count += deleted
    return archived_count, nonterminal


def _count_archive_candidates(
    live_conn: sqlite3.Connection, cutoff: str
) -> tuple[int, int]:
    eligible = 0
    nonterminal = 0
    cursor = live_conn.execute(
        "SELECT job_json FROM question_jobs WHERE created_at < ? ORDER BY created_at",
        (cutoff,),
    )
    while batch := cursor.fetchmany(BATCH_SIZE):
        for (job_json,) in batch:
            try:
                job = QuestionJob.model_validate_json(job_json)
            except Exception as exc:
                raise MaintenanceError("A job record could not be decoded") from exc
            if job.status in {"completed", "failed"}:
                eligible += 1
            else:
                nonterminal += 1
    return eligible, nonterminal


def _count_expired_archive_rows(archive_dir: Path, cutoff: str) -> int:
    expired = 0
    if not archive_dir.exists():
        return expired
    for archive_path in archive_dir.glob("jobs-*.sqlite3"):
        conn = _open_readonly_database(archive_path)
        try:
            _check_database(conn)
            expired += conn.execute(
                "SELECT COUNT(*) FROM archived_question_jobs WHERE created_at < ?",
                (cutoff,),
            ).fetchone()[0]
        finally:
            conn.close()
    return expired


def _prune_archive_databases(archive_dir: Path, cutoff: str) -> int:
    removed = 0
    for archive_path in sorted(archive_dir.glob("jobs-*.sqlite3")):
        conn = sqlite3.connect(archive_path, timeout=30.0)
        try:
            _check_database(conn)
            cursor = conn.execute(
                "DELETE FROM archived_question_jobs WHERE created_at < ?", (cutoff,)
            )
            pruned = cursor.rowcount
            conn.commit()
            remaining = conn.execute(
                "SELECT COUNT(*) FROM archived_question_jobs"
            ).fetchone()[0]
            if pruned and remaining:
                conn.execute("VACUUM")
            _check_database(conn)
        finally:
            conn.close()
        removed += max(pruned, 0)
        if remaining == 0:
            archive_path.unlink(missing_ok=True)
    return removed


def archive_old_jobs(
    *, dry_run: bool = False, vacuum_live: bool = False
) -> tuple[int, int, int]:
    """Archive old terminal jobs, then expire old archived records."""
    source = Path(settings.questions_db_path)
    now = _utc_now()
    live_cutoff = _format_timestamp(
        now - timedelta(days=settings.questions_live_retention_days)
    )
    archive_cutoff = _format_timestamp(
        now
        - timedelta(
            days=(
                settings.questions_live_retention_days
                + settings.questions_archive_retention_days
            )
        )
    )
    archive_dir = Path(settings.questions_archive_dir)

    if dry_run:
        live_conn = _open_readonly_database(source)
        try:
            _check_database(live_conn)
            columns = {
                row[1]
                for row in live_conn.execute(
                    "PRAGMA table_info(question_jobs)"
                ).fetchall()
            }
            if not {"job_json", "created_at"} <= columns:
                raise MaintenanceError(
                    "Start the application once to migrate the database before dry-run"
                )
            eligible, nonterminal = _count_archive_candidates(live_conn, live_cutoff)
            expired = _count_expired_archive_rows(archive_dir, archive_cutoff)
            print(
                f"Would archive {eligible} terminal jobs, "
                f"expire {expired} archived rows, "
                f"and leave {nonterminal} nonterminal jobs in the live database."
            )
            return 0, 0, nonterminal
        finally:
            live_conn.close()

    store = SQLiteQuestionJobStore(str(source), timeout=30.0)
    try:
        store.verify_integrity()
    finally:
        store.close()

    live_conn = sqlite3.connect(source, timeout=30.0)
    try:
        backup_live_database(now)
        _ensure_private_directory(archive_dir)
        archived = 0
        nonterminal = 0
        last_created_at = ""
        last_request_id = ""
        while True:
            rows = live_conn.execute(
                "SELECT request_id, job_json, created_at FROM question_jobs "
                "WHERE created_at < ? AND (created_at > ? OR "
                "(created_at = ? AND request_id > ?)) "
                "ORDER BY created_at, request_id LIMIT ?",
                (
                    live_cutoff,
                    last_created_at,
                    last_created_at,
                    last_request_id,
                    BATCH_SIZE,
                ),
            ).fetchall()
            if not rows:
                break
            last_request_id, last_created_at = rows[-1][0], rows[-1][2]
            batch_archived, batch_nonterminal = _archive_batch(
                live_conn, archive_dir, rows
            )
            archived += batch_archived
            nonterminal += batch_nonterminal

        pruned = _prune_archive_databases(archive_dir, archive_cutoff)
        if vacuum_live and archived:
            live_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            live_conn.execute("VACUUM")
        _check_database(live_conn)
        return archived, pruned, nonterminal
    finally:
        live_conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "backup-live", help="Snapshot the live database with GFS rotation"
    )
    commands.add_parser(
        "backup-archives", help="Snapshot changed monthly archive databases"
    )
    restore = commands.add_parser(
        "restore-check", help="Restore a backup to a temporary database and validate it"
    )
    restore.add_argument(
        "snapshot",
        nargs="?",
        type=Path,
        help="Snapshot to check; defaults to the newest live snapshot",
    )
    archive = commands.add_parser(
        "archive", help="Archive and expire terminal job rows"
    )
    archive.add_argument(
        "--dry-run",
        action="store_true",
        help="Count eligible rows without changing data",
    )
    archive.add_argument(
        "--vacuum-live",
        action="store_true",
        help="Rebuild the live database after archival to return free pages to the OS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "backup-live":
            path = backup_live_database()
            print(f"Created live database snapshot: {path}")
        elif args.command == "backup-archives":
            count = backup_archives()
            print(f"Created snapshots for {count} archive databases")
        elif args.command == "restore-check":
            count = verify_restore(args.snapshot)
            print(f"Restore check passed; snapshot contains {count} live job rows")
        else:
            archived, pruned, nonterminal = archive_old_jobs(
                dry_run=args.dry_run,
                vacuum_live=args.vacuum_live,
            )
            if not args.dry_run:
                print(
                    f"Archived {archived} jobs and pruned "
                    f"{pruned} expired archive rows; "
                    f"left {nonterminal} nonterminal live rows"
                )
    except (MaintenanceError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Question-job maintenance failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
