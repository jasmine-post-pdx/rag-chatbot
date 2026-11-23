"""Acceptance tests for deterministic context-budget allocation."""

import logging

import pytest

from app.domain.context import BudgetRequest, RankedChunk
from app.domain.errors import ContextBudgetExceededError
from app.services.token_budget import (
    ApproximateTokenCounter,
    allocate_context_budget,
    assert_allocation_within_budget,
    log_budget_allocation,
)


class ExactTokenCounter:
    """Count each character as one token for boundary-sensitive tests.

    :ivar is_estimate: Always ``False`` so tests can assert exact accounting."""

    is_estimate = False

    def count(self, text: str) -> int:
        """Return ``len(text)`` as the token count.

        :param text: Text to measure.
        :returns: Character length treated as a token count."""
        return len(text)


def _chunk(chunk_id: str, text: str, *, source_id: str = "src") -> RankedChunk:
    """Build a ranked chunk for budget tests.

    :param chunk_id: Chunk identifier preserved on selection.
    :param text: Chunk body counted against the retrieval budget.
    :param source_id: Source document identifier.
    :returns: A ``RankedChunk`` instance."""
    return RankedChunk(source_id=source_id, chunk_id=chunk_id, text=text)


def test_short_question_selects_ranked_chunks_until_budget_fills():
    """Select chunks in order for a short question until capacity is used."""
    counter = ExactTokenCounter()
    # framing = len("Evidence:\n") + len("\n\nQuestion:\n") = 10 + 12 = 22
    # fixed = question(5) + output(10) + margin(0) = 15
    # window 40 -> available retrieval = 25
    # first chunk needs 22 framing + 2 text = 24; remaining 1; second chunk needs sep(2)+2 > 1
    request = BudgetRequest(
        question="hello",
        ranked_chunks=[
            _chunk("c1", "aa"),
            _chunk("c2", "bb"),
            _chunk("c3", "cc"),
        ],
    )

    allocation = allocate_context_budget(
        request,
        counter=counter,
        context_window_tokens=40,
        output_reserve_tokens=10,
        safety_margin_tokens=0,
    )

    assert allocation.eligible_count == 3
    assert allocation.selected_count == 1
    assert allocation.discarded_count == 2
    assert [chunk.chunk_id for chunk in allocation.selected_chunks] == ["c1"]
    assert "Evidence:" in allocation.assembled_prompt
    assert "aa" in allocation.assembled_prompt
    assert allocation.assembled_prompt.endswith("hello")
    assert_allocation_within_budget(
        allocation,
        context_window_tokens=40,
        output_reserve_tokens=10,
        safety_margin_tokens=0,
    )


def test_oversized_chunk_is_skipped_so_later_chunks_can_fit():
    """Skip a chunk that cannot fit and continue with later smaller chunks."""
    counter = ExactTokenCounter()
    # framing=22; fixed = q(1)+out(5)+margin(0)=6; window=40; available=34
    # oversized needs 22+20=42 > 34 -> skip
    # small needs 22+2=24 <= 34 -> select
    request = BudgetRequest(
        question="q",
        ranked_chunks=[
            _chunk("too-big", "x" * 20),
            _chunk("fits", "ok"),
        ],
    )

    allocation = allocate_context_budget(
        request,
        counter=counter,
        context_window_tokens=40,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )

    assert allocation.selected_count == 1
    assert allocation.discarded_count == 1
    assert allocation.selected_chunks[0].chunk_id == "fits"
    assert allocation.selected_chunks[0].source_id == "src"


def test_long_history_reduces_retrieval_capacity():
    """Long history lowers available retrieval and can discard chunks."""
    counter = ExactTokenCounter()
    short = allocate_context_budget(
        BudgetRequest(
            question="q",
            history=[],
            ranked_chunks=[_chunk("c1", "ab"), _chunk("c2", "cd")],
        ),
        counter=counter,
        context_window_tokens=50,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )
    long_history = allocate_context_budget(
        BudgetRequest(
            question="q",
            history=["h" * 20],
            ranked_chunks=[_chunk("c1", "ab"), _chunk("c2", "cd")],
        ),
        counter=counter,
        context_window_tokens=50,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )

    assert short.available_retrieval_tokens > long_history.available_retrieval_tokens
    assert long_history.selected_count < short.selected_count
    assert long_history.discarded_count > short.discarded_count


def test_large_tool_schema_reduces_retrieval_capacity():
    """Large tool schemas reduce retrieval capacity like history does."""
    counter = ExactTokenCounter()
    without_tools = allocate_context_budget(
        BudgetRequest(
            question="q",
            ranked_chunks=[_chunk("c1", "ab"), _chunk("c2", "cd")],
        ),
        counter=counter,
        context_window_tokens=50,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )
    with_tools = allocate_context_budget(
        BudgetRequest(
            question="q",
            tool_schemas=['{"name":"' + ("t" * 20) + '"}'],
            ranked_chunks=[_chunk("c1", "ab"), _chunk("c2", "cd")],
        ),
        counter=counter,
        context_window_tokens=50,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )

    assert without_tools.available_retrieval_tokens > with_tools.available_retrieval_tokens
    assert with_tools.selected_count <= without_tools.selected_count


def test_boundary_includes_exact_fit_and_excludes_one_over():
    """Include a chunk that fits exactly and exclude one that is one token over."""
    counter = ExactTokenCounter()
    # framing=22; fixed=q(1)+out(5)+margin(0)=6; window=33; available=27
    # exact: 22 + 5 = 27
    exact = allocate_context_budget(
        BudgetRequest(question="q", ranked_chunks=[_chunk("exact", "12345")]),
        counter=counter,
        context_window_tokens=33,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )
    over = allocate_context_budget(
        BudgetRequest(question="q", ranked_chunks=[_chunk("over", "123456")]),
        counter=counter,
        context_window_tokens=33,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )

    assert exact.selected_count == 1
    assert exact.discarded_count == 0
    assert over.selected_count == 0
    assert over.discarded_count == 1
    assert over.assembled_prompt == "q"


def test_fixed_input_over_budget_raises_controlled_error():
    """Fail closed when instructions and question alone exceed the window."""
    counter = ExactTokenCounter()
    request = BudgetRequest(
        question="q" * 10,
        instructions="i" * 10,
        ranked_chunks=[_chunk("ignored", "chunk")],
    )

    with pytest.raises(ContextBudgetExceededError) as exc_info:
        allocate_context_budget(
            request,
            counter=counter,
            context_window_tokens=20,
            output_reserve_tokens=5,
            safety_margin_tokens=0,
        )

    error = exc_info.value
    assert error.estimated_fixed_tokens == 25
    assert error.context_window_tokens == 20
    assert "chunk" not in str(error)
    assert "iiii" not in str(error)


def test_assembled_input_stays_within_estimated_budget():
    """Verify the assembled prompt fits under the window after reserves."""
    counter = ApproximateTokenCounter()
    request = BudgetRequest(
        question="What is RAG?",
        instructions="Answer briefly.",
        history=["Earlier turn about embeddings."],
        tool_schemas=['{"name":"search_documents","parameters":{}}'],
        ranked_chunks=[
            _chunk("c1", "Retrieval augments generation with documents."),
            _chunk("c2", "Embeddings index chunks for similarity search."),
            _chunk("c3", "Citations should preserve chunk identifiers."),
        ],
    )

    allocation = allocate_context_budget(
        request,
        counter=counter,
        context_window_tokens=200,
        output_reserve_tokens=32,
        safety_margin_tokens=8,
    )

    assert allocation.is_estimate is True
    assert_allocation_within_budget(
        allocation,
        context_window_tokens=200,
        output_reserve_tokens=32,
        safety_margin_tokens=8,
    )


def test_budget_logging_omits_sensitive_content(caplog):
    """Log only safe budget metadata, never prompt or chunk text.

    :param caplog: Pytest fixture capturing log records."""
    counter = ExactTokenCounter()
    secret_question = "SECRET_QUESTION_TOKEN"
    secret_chunk = "SECRET_CHUNK_TOKEN"
    secret_history = "SECRET_HISTORY_TOKEN"
    secret_tool = "SECRET_TOOL_TOKEN"
    allocation = allocate_context_budget(
        BudgetRequest(
            question=secret_question,
            history=[secret_history],
            tool_schemas=[secret_tool],
            ranked_chunks=[_chunk("c1", secret_chunk)],
        ),
        counter=counter,
        context_window_tokens=200,
        output_reserve_tokens=5,
        safety_margin_tokens=0,
    )

    with caplog.at_level(logging.INFO, logger="app.services.token_budget"):
        log_budget_allocation(allocation, request_id="req-1", debug=True)

    assert any(record.message == "Context budget allocated" for record in caplog.records)
    joined = " ".join(
        f"{record.getMessage()} {record.__dict__}" for record in caplog.records
    )
    assert secret_question not in joined
    assert secret_chunk not in joined
    assert secret_history not in joined
    assert secret_tool not in joined
    record = next(
        item for item in caplog.records if item.message == "Context budget allocated"
    )
    assert record.request_id == "req-1"
    assert record.selected_count == allocation.selected_count
    assert record.available_retrieval_tokens == allocation.available_retrieval_tokens


def test_empty_chunks_returns_bare_question_prompt():
    """Keep the current generation input shape when no chunks are supplied."""
    allocation = allocate_context_budget(
        BudgetRequest(question="What is RAG?"),
        counter=ExactTokenCounter(),
        context_window_tokens=100,
        output_reserve_tokens=10,
        safety_margin_tokens=5,
    )

    assert allocation.eligible_count == 0
    assert allocation.selected_count == 0
    assert allocation.discarded_count == 0
    assert allocation.assembled_prompt == "What is RAG?"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"context_window_tokens": 0}, "context_window_tokens must be greater than 0"),
        ({"context_window_tokens": -1}, "context_window_tokens must be greater than 0"),
        ({"output_reserve_tokens": 0}, "output_reserve_tokens must be greater than 0"),
        ({"output_reserve_tokens": -1}, "output_reserve_tokens must be greater than 0"),
        ({"safety_margin_tokens": -1}, "safety_margin_tokens must be zero or greater"),
    ],
)
def test_allocate_context_budget_rejects_invalid_numeric_args(kwargs, match):
    """Reject non-positive window/output reserve and negative safety margin.

    :param kwargs: Invalid keyword override for ``allocate_context_budget``.
    :param match: Expected ``ValueError`` message fragment."""
    options = {
        "context_window_tokens": 40,
        "output_reserve_tokens": 5,
        "safety_margin_tokens": 0,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        allocate_context_budget(
            BudgetRequest(question="q"),
            counter=ExactTokenCounter(),
            **options,
        )
