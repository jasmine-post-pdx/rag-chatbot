"""Performance bounds for Step 8 retrieval-grounded answering."""

from __future__ import annotations

import time
from typing import Any

import pytest

from app.domain.model_outputs import AnswerClaim, AnswerV1
from app.llm import StructuredGenerationResult
from app.retrieval.document_store import DocumentChunk
from app.services.question_processor import GroundedQuestionProcessor
from app.services.token_budget import (
    ApproximateTokenCounter,
    ContextBudgetConfig,
    assert_allocation_within_budget,
)
from tests.support.overlap_store import TokenOverlapDocumentStore


class RecordingBudget(ContextBudgetConfig):
    """Record allocate kwargs so tests can inspect ranked_chunks and allocation.

    :ivar calls: Allocate kwargs captured in call order.
    :ivar allocations: Returned allocations in call order."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the budget and empty call logs.

        :param kwargs: Forwarded to ``ContextBudgetConfig``."""
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []
        self.allocations: list[Any] = []

    def allocate(self, question: str, **kwargs: Any):
        """Record kwargs, allocate, and store the result.

        :param question: User question text.
        :param kwargs: Additional allocate keyword arguments.
        :returns: The budget allocation."""
        self.calls.append({"question": question, **kwargs})
        allocation = super().allocate(question, **kwargs)
        self.allocations.append(allocation)
        return allocation


class FixedAnswerClient:
    """Return a fixed structured answer without tool turns.

    :ivar prompts: Prompts received, in call order."""

    def __init__(self, parsed: AnswerV1) -> None:
        """Initialize with a canned structured answer.

        :param parsed: Answer returned for every structured generation call."""
        self.prompts: list[str] = []
        self._parsed = parsed
        self.provider = "fake"
        self.model = "fake-perf"

    async def generate_structured(self, prompt: str, output_model: type):
        """Record the prompt and return the canned structured result.

        :param prompt: Prompt text to record.
        :param output_model: Expected output model type.
        :returns: Structured result wrapping the canned answer."""
        assert output_model is AnswerV1
        self.prompts.append(prompt)
        return StructuredGenerationResult(
            parsed=self._parsed,
            provider=self.provider,
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            finish_status="completed",
        )

    async def generate(self, question: str):
        """Satisfy the LLMClient protocol without being used."""
        raise AssertionError("unused")

    async def generate_tool_turn(self, *args: Any, **kwargs: Any):
        """Grounded path must not open a tool loop."""
        raise AssertionError("unused")


def _many_chunks(count: int) -> list[DocumentChunk]:
    """Build many overlapping Depends-related chunks for top-k tests.

    :param count: Number of chunks to create.
    :returns: A list of ``DocumentChunk`` instances."""
    return [
        DocumentChunk(
            source_id=f"src-{index}",
            chunk_id=f"chunk-{index}",
            text=(
                f"Chunk {index}: FastAPI dependency injection uses Depends "
                f"to inject shared resources into path operations."
            ),
            framework="fastapi",
            language="python",
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_retrieval_top_k_cap_is_never_exceeded():
    """Processor never asks the store for more hits than retrieval_top_k."""
    store = TokenOverlapDocumentStore(_many_chunks(40), max_limit=50)
    top_k = 5
    client = FixedAnswerClient(
        AnswerV1(
            answer="FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["chunk-0"],
                )
            ],
            insufficient_evidence=False,
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=ContextBudgetConfig(
            counter=ApproximateTokenCounter(),
            context_window_tokens=128000,
            output_reserve_tokens=64,
            safety_margin_tokens=0,
        ),
        retrieval_top_k=top_k,
    )

    result = await processor.process(
        "req-topk",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert store.search_calls
    assert all(call["limit"] <= top_k for call in store.search_calls)
    assert result.retrieval.chunks_returned <= top_k


@pytest.mark.asyncio
async def test_large_corpus_grounded_path_stays_within_time_bound():
    """Retrieve + budget + fake generate over a large synthetic corpus stays fast."""
    chunks = _many_chunks(500)
    store = TokenOverlapDocumentStore(chunks, max_limit=50)
    client = FixedAnswerClient(
        AnswerV1(
            answer="FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["chunk-0"],
                )
            ],
            insufficient_evidence=False,
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=ContextBudgetConfig(
            counter=ApproximateTokenCounter(),
            context_window_tokens=128000,
            output_reserve_tokens=64,
            safety_margin_tokens=0,
        ),
        retrieval_top_k=10,
    )

    started = time.perf_counter()
    result = await processor.process(
        "req-large",
        "How does FastAPI inject dependencies with Depends?",
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"grounded path took {elapsed:.3f}s"
    assert result.citations
    assert result.retrieval.documents_considered >= 1


@pytest.mark.asyncio
async def test_assembled_prompt_stays_within_estimated_budget():
    """Grounded allocation keeps assembled prompt tokens under the window reserves."""
    store = TokenOverlapDocumentStore(_many_chunks(30), max_limit=30)
    window = 500
    output_reserve = 50
    safety = 10
    budget = RecordingBudget(
        counter=ApproximateTokenCounter(),
        context_window_tokens=window,
        output_reserve_tokens=output_reserve,
        safety_margin_tokens=safety,
    )
    client = FixedAnswerClient(
        AnswerV1(
            answer="FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["chunk-0"],
                )
            ],
            insufficient_evidence=False,
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=budget,
        retrieval_top_k=20,
    )

    await processor.process(
        "req-fit",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert budget.allocations
    allocation = budget.allocations[0]
    assert_allocation_within_budget(
        allocation,
        context_window_tokens=window,
        output_reserve_tokens=output_reserve,
        safety_margin_tokens=safety,
    )
    # Prompt recorded by the client should match the assembled budget prompt.
    assert client.prompts
    assert client.prompts[0] == allocation.assembled_prompt
