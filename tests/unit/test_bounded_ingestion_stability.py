"""Event-loop isolation checks for Step 9 bounded ingestion."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.domain.questions import ModelMetrics, QuestionResult, RetrievalMetrics
from app.retrieval.chunking import SourceDocument
from app.retrieval.ingestion import BoundedIngestionPipeline
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.services.question_workflow import QuestionWorkflow
from tests.unit.test_bounded_ingestion import RecordingAsyncEmbedder, _source


class BlockingUpsertStore(InMemoryVectorDocumentStore):
    """Block the first ``upsert_chunks`` call on a worker thread until released."""

    def __init__(self, *args, started: threading.Event, release: threading.Event, **kwargs):
        """Wire barrier events around upsert.

        :param started: Set when upsert enters the blocking section.
        :param release: Cleared until the test allows upsert to finish.
        :param args: Forwarded to the in-memory store.
        :param kwargs: Forwarded to the in-memory store."""
        super().__init__(*args, **kwargs)
        self._started = started
        self._release = release
        self._blocked_once = False

    def upsert_chunks(self, chunks):
        """Block once so the event-loop probe can run.

        :param chunks: Embedded chunks to upsert.
        :returns: Whatever the base store returns."""
        if not self._blocked_once:
            self._blocked_once = True
            self._started.set()
            assert self._release.wait(timeout=2.0)
        return super().upsert_chunks(chunks)


@pytest.mark.asyncio
async def test_blocked_upsert_during_ingest_does_not_freeze_event_loop():
    """Sync store upsert offloaded during ingest leaves the loop responsive."""
    started = threading.Event()
    release = threading.Event()
    embedder = RecordingAsyncEmbedder(delay=0.0)
    store = BlockingUpsertStore(embedder, max_limit=10, started=started, release=release)
    pipeline = BoundedIngestionPipeline(store, embedder, concurrency=1)

    ingest_task = asyncio.create_task(
        pipeline.ingest_documents(
            [_source("blocked", "FastAPI Depends injects shared resources.")]
        )
    )
    entered = await asyncio.to_thread(started.wait, 0.5)
    if not entered:
        # Surface stub/early failures (e.g. NotImplementedError) immediately.
        await ingest_task
        pytest.fail("upsert never started during bounded ingest")

    probe_started = time.perf_counter()
    await asyncio.sleep(0.05)
    probe_elapsed = time.perf_counter() - probe_started
    assert probe_elapsed < 0.2

    release.set()
    result = await ingest_task
    assert result.sources_succeeded == 1


@pytest.mark.asyncio
async def test_slow_embed_does_not_block_question_workflow_probe(store_factory):
    """While ingest awaits a slow embed, a question workflow can still progress.

    :param store_factory: Fixture that creates a SQLite job store."""
    hold = asyncio.Event()
    entered = asyncio.Event()
    embedder = RecordingAsyncEmbedder(delay=0.0, hold=hold, entered=entered)
    store = InMemoryVectorDocumentStore(embedder, max_limit=10)
    pipeline = BoundedIngestionPipeline(store, embedder, concurrency=1)

    class FastProcessor:
        """Return immediately so the workflow probe measures loop availability."""

        async def process(self, question_id, question, *, request_id=None):
            """Return a completed result without awaiting a provider.

            :param question_id: Identifier of the question job.
            :param question: User question text.
            :param request_id: Optional HTTP correlation ID.
            :returns: A synthetic completed result."""
            return QuestionResult(
                question_id=question_id,
                answer=question,
                citations=[],
                retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
                model=ModelMetrics(
                    provider="test", name="test", input_tokens=0, output_tokens=0
                ),
            )

    job_store = store_factory()
    workflow = QuestionWorkflow(
        job_store, FastProcessor(), question_id_factory=lambda: "q-probe"
    )

    ingest_task = asyncio.create_task(
        pipeline.ingest_documents(
            [
                SourceDocument(
                    source_id="slow",
                    title="Slow",
                    text="Slow embedding document about FastAPI routing.",
                )
            ]
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.5)
    except TimeoutError:
        await ingest_task
        pytest.fail("embed hold never entered during bounded ingest")

    probe_started = time.perf_counter()
    question = await workflow.submit_question("Is the loop free?")
    probe_elapsed = time.perf_counter() - probe_started

    hold.set()
    ingest_result = await ingest_task

    assert question.answer == "Is the loop free?"
    assert probe_elapsed < 0.5
    assert ingest_result.sources_succeeded == 1
