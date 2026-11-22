"""Stability checks for async generation and thread-offloaded SQLite."""

import asyncio
import threading
import time

import pytest

from app.domain.questions import ModelMetrics, QuestionResult, RetrievalMetrics
from app.llm import FakeLLMClient, StructuredGenerationResult
from app.services.question_processor import LLMQuestionProcessor
from app.services.question_workflow import QuestionWorkflow
from app.stores.sqlite import SQLiteQuestionJobStore


def _result(question_id: str, answer: str = "ok") -> QuestionResult:
    """Build a minimal completed result for stability tests.

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


@pytest.mark.asyncio
async def test_blocked_sqlite_lookup_does_not_freeze_event_loop(store_factory):
    """A slow store call offloaded with ``to_thread`` leaves the loop responsive.

    :param store_factory: Fixture that creates a SQLite job store."""
    started = threading.Event()
    release = threading.Event()

    class BlockingStore(SQLiteQuestionJobStore):
        """Block the first completed-job insert until released by the test."""

        def create_completed_job(self, question_id, result, **kwargs):
            """Signal that the thread entered the store, then wait for release.

            :param question_id: Job identifier being persisted.
            :param result: Completed result to store.
            :param **kwargs: Idempotency metadata forwarded to the base store.
            :returns: The persisted completed job."""
            started.set()
            assert release.wait(timeout=2.0)
            return super().create_completed_job(question_id, result, **kwargs)

    class FastProcessor:
        """Return immediately so the workflow reaches the blocking store write."""

        async def process(self, question_id, question, *, request_id=None):
            """Return a completed result without awaiting a provider.

            :param question_id: Identifier of the question job.
            :param question: User question text.
            :param request_id: Optional HTTP correlation ID.
            :returns: A synthetic completed result."""
            return _result(question_id, answer=question)

    store = store_factory(store_type=BlockingStore)
    workflow = QuestionWorkflow(store, FastProcessor(), question_id_factory=lambda: "req-1")

    submit_task = asyncio.create_task(workflow.submit_question("What is RAG?"))
    assert await asyncio.to_thread(started.wait, 2.0)

    probe_started = time.perf_counter()
    await asyncio.sleep(0.05)
    probe_elapsed = time.perf_counter() - probe_started
    assert probe_elapsed < 0.2

    release.set()
    result = await submit_task
    assert result.answer == "What is RAG?"
    assert store.get_job("req-1").status == "completed"


@pytest.mark.asyncio
async def test_cancelled_generation_does_not_leak_store_lock(store_factory):
    """Cancel during ``await generate_structured`` and still reuse the store afterward.

    Verified behavior: cancelling the submit task before persistence leaves no
    terminal job for that ID, and a later submission can still write through the
    same store. The shared generation client lifecycle is owned by lifespan and
    is not closed by request cancellation.

    :param store_factory: Fixture that creates a SQLite job store."""
    entered = asyncio.Event()
    hold = asyncio.Event()

    class HoldingClient:
        """Pause inside generate_structured until the test cancels or releases it."""

        provider = "fake"
        model = "hold"

        async def generate_structured(self, prompt: str, output_model: type):
            """Wait until released, then return a deterministic structured answer.

            :param prompt: Prompt that would be answered.
            :param output_model: Expected structured output type.
            :returns: A structured generation result after the hold is cleared."""
            entered.set()
            await hold.wait()
            return StructuredGenerationResult(
                parsed=output_model(
                    answer=f"Late answer to: {prompt}",
                    claims=[],
                    insufficient_evidence=True,
                ),
                provider="fake",
                model="hold",
                input_tokens=0,
                output_tokens=0,
            )

    store = store_factory(":memory:")
    workflow = QuestionWorkflow(
        store,
        LLMQuestionProcessor(HoldingClient()),
        question_id_factory=lambda: "req-cancel",
    )

    task = asyncio.create_task(workflow.submit_question("Cancel me"))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    recovered = QuestionWorkflow(
        store,
        LLMQuestionProcessor(FakeLLMClient(model="recovered")),
        question_id_factory=lambda: "req-recovered",
    )
    result = await recovered.submit_question("Recovered")
    assert result.answer == "Fake answer to: Recovered"
    assert store.get_job("req-recovered").status == "completed"
    assert store.get_job("req-cancel") is None


@pytest.mark.asyncio
async def test_blocked_readiness_check_does_not_freeze_event_loop(store_factory):
    """A blocked ``is_ready`` offloaded with ``to_thread`` leaves the loop free.

    This mirrors the readiness route's ``await asyncio.to_thread(store.is_ready)``
    path without requiring concurrent ASGI lifespan clients.

    :param store_factory: Fixture that creates a SQLite job store."""
    started = threading.Event()
    release = threading.Event()

    class BlockingReadyStore(SQLiteQuestionJobStore):
        """Block ``is_ready`` until the probe completes."""

        def is_ready(self) -> bool:
            """Signal entry, wait for release, then delegate to the base store.

            :returns: Whether the underlying store is ready."""
            started.set()
            assert release.wait(timeout=2.0)
            return super().is_ready()

    store = store_factory(store_type=BlockingReadyStore)
    ready_task = asyncio.create_task(asyncio.to_thread(store.is_ready))
    assert await asyncio.to_thread(started.wait, 2.0)

    probe_started = time.perf_counter()
    await asyncio.sleep(0.05)
    assert time.perf_counter() - probe_started < 0.2

    release.set()
    assert await ready_task is True
