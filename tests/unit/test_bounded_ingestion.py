"""Unit tests for Step 9 bounded concurrent ingestion."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import SourceDocument
from app.retrieval.ingestion import (
    BoundedIngestionPipeline,
    ConcurrentIngestResult,
    IngestSourceFailure,
)
from app.retrieval.memory_vector import InMemoryVectorDocumentStore


class RecordingAsyncEmbedder(FakeEmbeddingClient):
    """Fake embedder that records in-flight async embed calls.

    :ivar max_in_flight: Peak concurrent ``aembed_documents`` calls observed.
    :ivar call_count: Total ``aembed_documents`` invocations.
    :ivar _delay: Artificial await delay per embed call.
    :ivar _fail_texts: Substrings that trigger an embed failure when present.
    :ivar _hold: Optional event that blocks embed until set.
    :ivar _entered: Optional event set when the first embed call starts.
    :ivar _lock: Protects in-flight counters.
    :ivar _in_flight: Current concurrent embed calls."""

    def __init__(
        self,
        *,
        dimensions: int = 16,
        model: str = "fake-bounded",
        delay: float = 0.05,
        fail_texts: frozenset[str] | None = None,
        hold: asyncio.Event | None = None,
        entered: asyncio.Event | None = None,
    ) -> None:
        """Configure recording and optional failure/hold behavior.

        :param dimensions: Embedding width.
        :param model: Reported model name.
        :param delay: Seconds to await inside each ``aembed_documents`` call.
        :param fail_texts: If any document text contains one of these, raise.
        :param hold: When set, embed waits until this event is set.
        :param entered: Set when the first embed call begins."""
        super().__init__(dimensions=dimensions, model=model)
        self.max_in_flight = 0
        self.call_count = 0
        self._delay = delay
        self._fail_texts = fail_texts or frozenset()
        self._hold = hold
        self._entered = entered
        self._lock = threading.Lock()
        self._in_flight = 0
        self._closed = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sync embed used by the in-memory store for search.

        :param texts: Document texts.
        :returns: Embeddings."""
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Sync query embed for search.

        :param text: Query text.
        :returns: Embedding."""
        return super().embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Record concurrency, optionally fail or hold, then embed.

        :param texts: Document texts for one source batch.
        :returns: Embeddings.
        :raises RuntimeError: When a configured fail substring is present."""
        if self._entered is not None and not self._entered.is_set():
            self._entered.set()
        with self._lock:
            self._in_flight += 1
            self.call_count += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self._hold is not None:
                await self._hold.wait()
            if self._delay:
                await asyncio.sleep(self._delay)
            for text in texts:
                for marker in self._fail_texts:
                    if marker in text:
                        raise RuntimeError(f"embed failed for marker={marker}")
            return [self.embed_query(text) if text.strip() else self.embed_query("x") for text in texts]
        finally:
            with self._lock:
                self._in_flight -= 1

    async def aembed_query(self, text: str) -> list[float]:
        """Async query embed.

        :param text: Query text.
        :returns: Embedding."""
        return self.embed_query(text)

    async def aclose(self) -> None:
        """Mark the client closed."""
        self._closed = True

    @property
    def in_flight(self) -> int:
        """Return the current in-flight embed count."""
        with self._lock:
            return self._in_flight


def _source(source_id: str, text: str, **kwargs) -> SourceDocument:
    """Build a source document for bounded ingestion tests.

    :param source_id: Stable source identifier.
    :param text: Body text.
    :param kwargs: Optional framework/language/title overrides.
    :returns: A ``SourceDocument``."""
    return SourceDocument(
        source_id=source_id,
        title=kwargs.get("title", source_id),
        text=text,
        framework=kwargs.get("framework"),
        language=kwargs.get("language", "python"),
        metadata=kwargs.get("metadata", {"path": f"data/raw/{source_id}.md"}),
    )


def _pipeline(
    *,
    concurrency: int = 2,
    delay: float = 0.05,
    fail_texts: frozenset[str] | None = None,
    hold: asyncio.Event | None = None,
    entered: asyncio.Event | None = None,
    timeout_seconds: float | None = None,
    source_timeout_seconds: float | None = None,
    chunk_size: int = 200,
) -> tuple[BoundedIngestionPipeline, InMemoryVectorDocumentStore, RecordingAsyncEmbedder]:
    """Build a bounded pipeline with a recording async embedder.

    :returns: Pipeline, store, and embedder."""
    embedder = RecordingAsyncEmbedder(
        delay=delay,
        fail_texts=fail_texts,
        hold=hold,
        entered=entered,
    )
    store = InMemoryVectorDocumentStore(embedder, max_limit=20)
    pipeline = BoundedIngestionPipeline(
        store,
        embedder,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        source_timeout_seconds=source_timeout_seconds,
        chunk_size=chunk_size,
        chunk_overlap=20,
    )
    return pipeline, store, embedder


@pytest.mark.asyncio
async def test_ingest_documents_upserts_multiple_sources():
    """Bounded ingest upserts several sources and returns concurrent result counts."""
    pipeline, store, _embedder = _pipeline(concurrency=2, delay=0.01)
    sources = [
        _source("fastapi-guide", "FastAPI Depends injects shared resources into routes."),
        _source("rag-budget", "Context budgeting reserves tokens for output reserve."),
    ]

    result = await pipeline.ingest_documents(sources)

    assert isinstance(result, ConcurrentIngestResult)
    assert result.sources_processed == 2
    assert result.sources_succeeded == 2
    assert result.sources_failed == 0
    assert result.failures == []
    assert result.chunks_upserted >= 2
    assert store.search_documents("Depends injects", limit=5)
    assert store.search_documents("budgeting reserves", limit=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [2, 5])
async def test_concurrency_cap_is_never_exceeded(concurrency: int):
    """In-flight embed calls never exceed the configured concurrency cap.

    :param concurrency: Cap under test."""
    pipeline, _store, embedder = _pipeline(concurrency=concurrency, delay=0.08)
    sources = [
        _source(f"src-{i}", f"Document {i} discusses FastAPI routing and Depends.")
        for i in range(concurrency * 3)
    ]

    result = await pipeline.ingest_documents(sources)

    assert result.sources_succeeded == len(sources)
    assert embedder.max_in_flight <= concurrency
    assert embedder.max_in_flight >= min(concurrency, len(sources))


@pytest.mark.asyncio
async def test_partial_failure_continues_other_sources():
    """One embed failure is recorded; sibling sources still succeed."""
    marker = "FAIL_THIS_SOURCE"
    pipeline, store, _embedder = _pipeline(
        concurrency=2,
        delay=0.01,
        fail_texts=frozenset({marker}),
    )
    sources = [
        _source("ok-source", "Express middleware runs in order for each request."),
        _source("bad-source", f"Body containing {marker} should fail embedding."),
        _source("also-ok", "FastAPI lifespan creates shared clients on startup."),
    ]

    result = await pipeline.ingest_documents(sources)

    assert result.sources_processed == 3
    assert result.sources_succeeded == 2
    assert result.sources_failed == 1
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert isinstance(failure, IngestSourceFailure)
    assert failure.source_id == "bad-source"
    assert failure.category in {"embed", "error", "provider"}
    assert marker not in failure.category
    assert marker not in str(failure)
    assert store.search_documents("middleware", limit=5)
    assert store.search_documents("lifespan creates", limit=5)
    assert store.get_document_chunk("bad-source:0") is None


@pytest.mark.asyncio
async def test_source_timeout_records_failure_and_allows_siblings():
    """An overdue source fails with timeout; faster siblings can still complete."""
    hold = asyncio.Event()
    entered = asyncio.Event()
    pipeline, store, embedder = _pipeline(
        concurrency=2,
        delay=0.01,
        hold=hold,
        entered=entered,
        source_timeout_seconds=0.05,
    )
    # First call will block on hold until timeout; release after starting so
    # subsequent sources (if any) can proceed once the timed-out slot frees.
    sources = [
        _source("slow-source", "This source will be held past the deadline."),
        _source("fast-source", "FastAPI Depends wires routers to services quickly."),
    ]

    async def release_after_timeout() -> None:
        await asyncio.sleep(0.2)
        hold.set()

    releaser = asyncio.create_task(release_after_timeout())
    try:
        result = await pipeline.ingest_documents(sources)
    finally:
        hold.set()
        await releaser

    assert result.sources_failed >= 1
    timeout_failures = [f for f in result.failures if f.category == "timeout"]
    assert timeout_failures
    assert any(f.source_id == "slow-source" for f in timeout_failures)
    # At least the non-held work should be able to land once released, or the
    # pipeline may have completed the fast source on a free slot.
    assert result.sources_succeeded + result.sources_failed == result.sources_processed
    assert embedder.in_flight == 0


@pytest.mark.asyncio
async def test_cancellation_stops_work_and_leaves_store_usable():
    """Cancelling mid-ingest raises CancelledError and allows a later ingest."""
    hold = asyncio.Event()
    entered = asyncio.Event()
    pipeline, store, embedder = _pipeline(
        concurrency=2,
        delay=0.01,
        hold=hold,
        entered=entered,
    )
    sources = [
        _source(f"src-{i}", f"Cancellation corpus document {i} about FastAPI.")
        for i in range(6)
    ]

    task = asyncio.create_task(pipeline.ingest_documents(sources))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Allow any stragglers to observe cancellation, then settle.
    hold.set()
    await asyncio.sleep(0.05)
    assert embedder.in_flight == 0

    later = await pipeline.ingest_documents(
        [_source("recovered", "Recovered ingest after cancellation still works.")]
    )
    assert later.sources_succeeded == 1
    assert store.search_documents("Recovered ingest", limit=5)


@pytest.mark.asyncio
async def test_cleanup_after_partial_failure_clears_in_flight():
    """After partial failure the embedder reports zero in-flight calls."""
    pipeline, _store, embedder = _pipeline(
        concurrency=3,
        delay=0.02,
        fail_texts=frozenset({"BOOM"}),
    )
    sources = [
        _source("a", "Alpha document about FastAPI routing."),
        _source("b", "Beta document BOOM should fail."),
        _source("c", "Gamma document about Express middleware."),
    ]

    result = await pipeline.ingest_documents(sources)

    assert result.sources_failed == 1
    assert embedder.in_flight == 0


@pytest.mark.asyncio
async def test_reingest_under_concurrency_replaces_stale_chunks_only():
    """Re-ingesting one source deletes its stale chunks without touching others."""
    pipeline, store, _embedder = _pipeline(concurrency=2, delay=0.01, chunk_size=80)
    original = _source(
        "rag-notes",
        "Context budgeting reserves tokens for instructions and output reserve.",
    )
    other = _source(
        "fastapi-guide",
        "FastAPI dependency injection wires routers to services with Depends.",
        framework="fastapi",
    )
    await pipeline.ingest_documents([original, other])
    original_ids = {
        hit.chunk_id for hit in store.search_documents("budgeting", limit=10)
    }
    assert original_ids

    updated = _source(
        "rag-notes",
        "Updated budgeting policy skips oversized chunks without silent truncation.",
    )
    second = await pipeline.ingest_documents([updated, other])
    assert second.sources_succeeded == 2
    assert second.sources_deleted_before_upsert >= 1

    remaining = store.search_documents("budgeting", limit=10)
    assert "silent truncation" in " ".join(hit.text for hit in remaining)
    assert not any("output reserve" in hit.text for hit in remaining)
    fastapi_hits = store.search_documents("dependency injection", limit=5)
    assert fastapi_hits
    assert all(hit.source_id == "fastapi-guide" for hit in fastapi_hits)


@pytest.mark.asyncio
async def test_ingest_empty_corpus_is_a_no_op():
    """An empty source list upserts nothing."""
    pipeline, store, _embedder = _pipeline(delay=0.0)

    result = await pipeline.ingest_documents([])

    assert result.sources_processed == 0
    assert result.sources_succeeded == 0
    assert result.sources_failed == 0
    assert result.chunks_upserted == 0
    assert store.search_documents("anything", limit=5) == []


@pytest.mark.asyncio
async def test_ingest_empty_directory_is_a_no_op(tmp_path: Path):
    """An empty directory returns a zero concurrent result."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pipeline, store, _embedder = _pipeline(delay=0.0)

    result = await pipeline.ingest_directory(raw)

    assert result.sources_processed == 0
    assert result.chunks_upserted == 0
    assert store.search_documents("anything", limit=5) == []


@pytest.mark.asyncio
async def test_ingest_directory_loads_markdown_files(tmp_path: Path):
    """Directory ingest loads ``.md`` stems as source_id and ignores other files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fastapi-guide.md").write_text(
        "---\ntitle: FastAPI Guide\nframework: fastapi\nlanguage: python\n---\n"
        "FastAPI Depends injects shared resources into path operations.\n",
        encoding="utf-8",
    )
    (raw / "notes.txt").write_text("ignored extension", encoding="utf-8")

    pipeline, store, _embedder = _pipeline(delay=0.01)
    result = await pipeline.ingest_directory(raw)

    assert result.sources_processed == 1
    assert result.sources_succeeded == 1
    assert result.chunks_upserted >= 1
    hits = store.search_documents("Depends injects", limit=5)
    assert hits
    assert hits[0].source_id == "fastapi-guide"
    assert hits[0].title == "FastAPI Guide"


@pytest.mark.asyncio
async def test_overall_timeout_fails_job_safely():
    """An overall job deadline produces timeout failures without hanging forever."""
    hold = asyncio.Event()
    pipeline, _store, embedder = _pipeline(
        concurrency=2,
        delay=0.01,
        hold=hold,
        timeout_seconds=0.05,
    )
    sources = [
        _source(f"src-{i}", f"Overall timeout document {i} about FastAPI Depends.")
        for i in range(4)
    ]

    started = time.perf_counter()
    result = None
    message = ""
    try:
        result = await pipeline.ingest_documents(sources)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        message = str(exc)
    finally:
        hold.set()

    elapsed = time.perf_counter() - started
    assert elapsed < 2.0
    if result is not None:
        assert result.sources_failed >= 1
        assert any(f.category == "timeout" for f in result.failures)
        message = " ".join(f.category for f in result.failures)
    assert "SECRET_BODY" not in message
    await asyncio.sleep(0.05)
    assert embedder.in_flight == 0
