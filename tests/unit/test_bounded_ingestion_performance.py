"""Performance bounds for Step 9 bounded concurrent ingestion."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.retrieval.chunking import SourceDocument
from app.retrieval.ingestion import BoundedIngestionPipeline
from tests.unit.test_bounded_ingestion import RecordingAsyncEmbedder, _pipeline, _source


@pytest.mark.asyncio
async def test_burst_of_sources_never_exceeds_concurrency_cap():
    """A burst of 20+ sources keeps in-flight embeds at or below the cap."""
    concurrency = 3
    pipeline, _store, embedder = _pipeline(concurrency=concurrency, delay=0.04)
    sources = [
        _source(f"burst-{i}", f"Burst document {i} covers FastAPI Depends and routing.")
        for i in range(24)
    ]

    result = await pipeline.ingest_documents(sources)

    assert result.sources_succeeded == 24
    assert embedder.max_in_flight <= concurrency
    assert embedder.call_count >= 1


@pytest.mark.asyncio
async def test_higher_concurrency_finishes_faster_than_serial():
    """With artificial delay, concurrency=4 beats concurrency=1 on the same batch."""
    sources = [
        _source(f"perf-{i}", f"Performance document {i} about context budgeting tokens.")
        for i in range(8)
    ]
    delay = 0.05

    serial, _s1, _e1 = _pipeline(concurrency=1, delay=delay)
    parallel, _s2, _e2 = _pipeline(concurrency=4, delay=delay)

    started = time.perf_counter()
    serial_result = await serial.ingest_documents(sources)
    serial_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    parallel_result = await parallel.ingest_documents(sources)
    parallel_elapsed = time.perf_counter() - started

    assert serial_result.sources_succeeded == 8
    assert parallel_result.sources_succeeded == 8
    assert parallel_elapsed < serial_elapsed * 0.75
    assert parallel_elapsed < 1.5


@pytest.mark.asyncio
async def test_large_sources_complete_within_wall_clock_bound():
    """Large synthetic bodies still finish under a generous fake-embedder bound."""
    big = "FastAPI Depends injects resources. " * 400
    sources = [_source(f"large-{i}", big) for i in range(6)]
    pipeline, _store, _embedder = _pipeline(concurrency=3, delay=0.01, chunk_size=500)

    started = time.perf_counter()
    result = await pipeline.ingest_documents(sources)
    elapsed = time.perf_counter() - started

    assert result.sources_succeeded == 6
    assert result.chunks_upserted >= 6
    assert elapsed < 5.0, f"large ingest too slow: {elapsed:.3f}s"


@pytest.mark.parametrize("concurrency", [0, 11, -1])
def test_invalid_concurrency_rejected_by_runner(concurrency: int):
    """Runner construction rejects concurrency outside 1–10.

    :param concurrency: Invalid concurrency value."""
    embedder = RecordingAsyncEmbedder(delay=0.0)
    from app.retrieval.memory_vector import InMemoryVectorDocumentStore

    store = InMemoryVectorDocumentStore(embedder, max_limit=5)
    with pytest.raises(ValueError, match="concurrency"):
        BoundedIngestionPipeline(store, embedder, concurrency=concurrency)


@pytest.mark.parametrize("value", ["0", "11"])
def test_invalid_concurrency_rejected_by_settings(monkeypatch, value: str):
    """Settings validation rejects ``INGESTION_CONCURRENCY`` outside 1–10.

    :param monkeypatch: Pytest fixture for env overrides.
    :param value: Invalid env string."""
    monkeypatch.setenv("INGESTION_CONCURRENCY", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.asyncio
async def test_empty_and_single_source_are_fast():
    """Trivial batches complete quickly under the bounded runner."""
    pipeline, _store, _embedder = _pipeline(concurrency=5, delay=0.0)

    started = time.perf_counter()
    empty = await pipeline.ingest_documents([])
    single = await pipeline.ingest_documents(
        [SourceDocument(source_id="one", title="One", text="FastAPI routing basics.")]
    )
    elapsed = time.perf_counter() - started

    assert empty.sources_processed == 0
    assert single.sources_succeeded == 1
    assert elapsed < 1.0
