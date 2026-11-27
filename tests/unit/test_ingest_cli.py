"""CLI contract tests for Step 9 bounded concurrent ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import app.retrieval.ingestion as ingestion_mod
from app.retrieval.ingestion import ConcurrentIngestResult, IngestSourceFailure
from scripts import ingest as ingest_script


def _write_corpus(raw: Path) -> None:
    """Write a tiny markdown corpus for CLI runs.

    :param raw: Destination directory for ``.md`` sources."""
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "fastapi-guide.md").write_text(
        "---\ntitle: FastAPI Guide\nframework: fastapi\nlanguage: python\n---\n"
        "FastAPI dependency injection wires routers to services with Depends.\n",
        encoding="utf-8",
    )
    (raw / "rag-budget.md").write_text(
        "---\ntitle: RAG Context Budget\nlanguage: python\n---\n"
        "Context budgeting reserves tokens for instructions and a safety margin.\n",
        encoding="utf-8",
    )


def test_cli_constructs_bounded_pipeline_with_settings_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``main`` builds ``BoundedIngestionPipeline`` using ``ingestion_concurrency``.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest fixture for isolation."""
    corpus = tmp_path / "raw"
    _write_corpus(corpus)

    constructed: list[dict] = []

    class RecordingBoundedPipeline:
        """Record constructor kwargs and return a canned async result."""

        def __init__(self, store, embedder, **kwargs):
            """Capture construction for assertions.

            :param store: Vector store passed by the CLI.
            :param embedder: Embedding client passed by the CLI.
            :param kwargs: Remaining pipeline configuration."""
            constructed.append({"store": store, "embedder": embedder, **kwargs})

        async def ingest_directory(self, path: Path) -> ConcurrentIngestResult:
            """Return a successful concurrent ingest summary.

            :param path: Directory being ingested.
            :returns: Canned concurrent result."""
            assert path == corpus
            return ConcurrentIngestResult(
                sources_processed=2,
                chunks_upserted=2,
                sources_deleted_before_upsert=0,
                sources_succeeded=2,
                sources_failed=0,
                failures=[],
            )

    monkeypatch.setattr(ingest_script.settings, "corpus_raw_dir", str(corpus))
    monkeypatch.setattr(ingest_script.settings, "vector_store", "memory")
    monkeypatch.setattr(ingest_script.settings, "embedding_provider", "fake")
    monkeypatch.setattr(ingest_script.settings, "ingestion_concurrency", 7)
    monkeypatch.setattr(ingest_script.settings, "ingestion_timeout_seconds", 60.0)
    monkeypatch.setattr(
        ingest_script.settings, "ingestion_source_timeout_seconds", 12.0
    )
    monkeypatch.setattr(ingestion_mod, "BoundedIngestionPipeline", RecordingBoundedPipeline)
    monkeypatch.setattr(
        ingest_script, "BoundedIngestionPipeline", RecordingBoundedPipeline, raising=False
    )

    code = ingest_script.main(["--path", str(corpus)])

    assert code == 0
    assert len(constructed) == 1
    assert constructed[0]["concurrency"] == 7
    assert constructed[0]["timeout_seconds"] == 60.0
    assert constructed[0]["source_timeout_seconds"] == 12.0


def test_cli_logs_failure_counts_on_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """CLI summary logging includes succeeded/failed counts for partial runs.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest fixture for isolation.
    :param caplog: Pytest log capture fixture."""
    corpus = tmp_path / "raw"
    _write_corpus(corpus)

    class PartialPipeline:
        """Return a partial-failure concurrent result."""

        def __init__(self, *args, **kwargs):
            """Ignore construction details for this log-focused test."""

        async def ingest_directory(self, path: Path) -> ConcurrentIngestResult:
            """Return one success and one failure.

            :param path: Directory being ingested.
            :returns: Partial concurrent result."""
            return ConcurrentIngestResult(
                sources_processed=2,
                chunks_upserted=1,
                sources_deleted_before_upsert=0,
                sources_succeeded=1,
                sources_failed=1,
                failures=[
                    IngestSourceFailure(source_id="rag-budget", category="embed")
                ],
            )

    monkeypatch.setattr(ingest_script.settings, "vector_store", "memory")
    monkeypatch.setattr(ingest_script.settings, "embedding_provider", "fake")
    monkeypatch.setattr(ingest_script.settings, "ingestion_concurrency", 5)
    monkeypatch.setattr(ingestion_mod, "BoundedIngestionPipeline", PartialPipeline)
    monkeypatch.setattr(
        ingest_script, "BoundedIngestionPipeline", PartialPipeline, raising=False
    )

    with caplog.at_level(logging.INFO):
        code = ingest_script.main(["--path", str(corpus)])

    assert code == 0
    messages = " ".join(record.getMessage() for record in caplog.records).casefold()
    assert "failed" in messages
    assert "1" in messages
    assert "top_secret" not in messages
