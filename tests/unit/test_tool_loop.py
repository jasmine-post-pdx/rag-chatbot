"""Unit tests for the bounded model → tool → observation → answer loop."""

import pytest

from app.domain.errors import StructuredOutputError
from app.domain.model_outputs import AnswerClaim, AnswerV1
from app.domain.tools import ToolCallProposal
from app.llm.generation import FakeLLMClient, StructuredGenerationResult, ToolTurnResult
from app.retrieval.document_store import DocumentChunk
from app.retrieval.memory import InMemoryDocumentStore
from app.tools.executor import SafeToolExecutor
from app.tools.loop import ToolLoop, ToolLoopResult


def _answer(text: str = "FastAPI uses Depends for injection.") -> AnswerV1:
    """Build a valid AnswerV1 for loop tests.

    :param text: Answer text.
    :returns: A structured answer with one cited claim."""
    return AnswerV1(
        answer=text,
        claims=[
            AnswerClaim(claim="Depends wires dependencies", citation_chunk_ids=["c1"])
        ],
        insufficient_evidence=False,
    )


def _store() -> InMemoryDocumentStore:
    """Return a tiny corpus for loop tests.

    :returns: In-memory document store."""
    return InMemoryDocumentStore(
        [
            DocumentChunk(
                source_id="fastapi-guide",
                chunk_id="c1",
                text="FastAPI dependency injection wires routers via Depends.",
                framework="fastapi",
                language="python",
            )
        ]
    )


def _executor(store: InMemoryDocumentStore | None = None, **kwargs) -> SafeToolExecutor:
    """Build a safe executor for loop tests.

    :param store: Optional document store override.
    :param kwargs: Executor configuration overrides.
    :returns: A ``SafeToolExecutor``."""
    defaults = {
        "store": store if store is not None else _store(),
        "max_result_chars": 4000,
        "call_timeout_seconds": 2.0,
        "search_max_limit": 5,
    }
    defaults.update(kwargs)
    return SafeToolExecutor(**defaults)


def _loop(
    client: FakeLLMClient,
    *,
    max_turns: int = 3,
    max_calls: int = 6,
    max_calls_per_turn: int = 3,
    executor: SafeToolExecutor | None = None,
) -> ToolLoop:
    """Build a tool loop for tests.

    :param client: Scripted fake LLM client.
    :param max_turns: Maximum model↔tool iterations.
    :param max_calls: Maximum total tool executions.
    :param max_calls_per_turn: Maximum tool calls accepted from one model turn.
    :param executor: Optional executor override.
    :returns: A ``ToolLoop`` instance."""
    return ToolLoop(
        client=client,
        executor=executor if executor is not None else _executor(),
        max_turns=max_turns,
        max_calls=max_calls,
        max_calls_per_turn=max_calls_per_turn,
        search_max_limit=5,
    )


@pytest.mark.asyncio
async def test_happy_path_search_then_final_answer():
    """Run model → search_documents → observation → AnswerV1."""
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-1",
                        name="search_documents",
                        arguments={"query": "Depends", "limit": 2},
                    )
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=_answer(),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                    finish_status="completed",
                ),
            ),
        ]
    )
    loop = _loop(client)

    result = await loop.run("How does FastAPI inject deps?", request_id="req-1")

    assert isinstance(result, ToolLoopResult)
    assert result.answer.answer.startswith("FastAPI")
    assert result.documents_considered >= 1
    assert result.chunks_returned >= 1
    assert result.tool_calls_executed == 1
    assert len(client.tool_turn_inputs) == 2
    # Second turn should include the function_call_output observation.
    second_input = client.tool_turn_inputs[1]
    assert isinstance(second_input, list)
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in second_input
    )


@pytest.mark.asyncio
async def test_multi_tool_turn_within_bounds():
    """Execute multiple tool calls in one turn, then finish."""
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-a",
                        name="search_documents",
                        arguments={"query": "Depends"},
                    ),
                    ToolCallProposal(
                        call_id="call-b",
                        name="get_document_chunk",
                        arguments={"chunk_id": "c1"},
                    ),
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=_answer("Done"),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        ]
    )
    loop = _loop(client)

    result = await loop.run("Explain Depends", request_id="req-2")

    assert result.tool_calls_executed == 2
    assert result.answer.answer == "Done"


@pytest.mark.asyncio
async def test_loop_bound_exceeded_fails_closed():
    """Stop runaway loops when max_turns is exceeded."""
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id=f"call-{i}",
                        name="search_documents",
                        arguments={"query": "x"},
                    )
                ],
            )
            for i in range(5)
        ]
    )
    loop = _loop(client, max_turns=2, max_calls=10)

    with pytest.raises(StructuredOutputError) as exc_info:
        await loop.run("loop forever", request_id="req-3")

    assert exc_info.value.category == "unsupported"


@pytest.mark.asyncio
async def test_max_calls_per_turn_enforced():
    """Cap tool calls accepted from a single model turn."""
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id=f"call-{i}",
                        name="search_documents",
                        arguments={"query": "x"},
                    )
                    for i in range(5)
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=_answer("Capped"),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        ]
    )
    loop = _loop(client, max_calls_per_turn=2, max_calls=10)

    result = await loop.run("many tools", request_id="req-4")

    assert result.tool_calls_executed == 2
    assert result.answer.answer == "Capped"


@pytest.mark.asyncio
async def test_invented_tool_becomes_observation_then_model_can_finish():
    """Continue after an unknown tool with a safe error observation."""
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-bad",
                        name="admin_shell",
                        arguments={"cmd": "id"},
                    )
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=AnswerV1(
                        answer="I cannot use that tool.",
                        claims=[],
                        insufficient_evidence=True,
                    ),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        ]
    )
    loop = _loop(client)

    result = await loop.run("hack the box", request_id="req-5")

    assert result.tool_calls_executed == 1
    assert result.answer.insufficient_evidence is True
    second_input = client.tool_turn_inputs[1]
    assert isinstance(second_input, list)
    outputs = [
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert outputs
    assert "unknown_tool" in outputs[0]["output"]


@pytest.mark.asyncio
async def test_executor_timeout_becomes_observation_not_crash():
    """Map executor timeouts to observations without crashing the loop."""
    import time

    class SlowStore:
        def search_documents(self, *args, **kwargs):
            time.sleep(0.2)
            return []

        def get_document_chunk(self, chunk_id: str):
            return None

    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-slow",
                        name="search_documents",
                        arguments={"query": "x"},
                    )
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=AnswerV1(
                        answer="Timed out looking that up.",
                        claims=[],
                        insufficient_evidence=True,
                    ),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        ]
    )
    loop = _loop(
        client,
        executor=_executor(store=SlowStore(), call_timeout_seconds=0.05),
    )

    result = await loop.run("slow path", request_id="req-6")

    assert result.answer.insufficient_evidence is True
    second_input = client.tool_turn_inputs[1]
    assert any(
        isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and "timeout" in item.get("output", "")
        for item in second_input
    )


@pytest.mark.asyncio
async def test_output_size_cap_end_to_end_through_loop():
    """Verify oversized tool content is capped before the next model turn."""
    huge = DocumentChunk(
        source_id="big",
        chunk_id="huge",
        text="Z" * 500,
    )
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-cap",
                        name="get_document_chunk",
                        arguments={"chunk_id": "huge"},
                    )
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=_answer("Got a truncated chunk."),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        ]
    )
    loop = _loop(
        client,
        executor=_executor(
            store=InMemoryDocumentStore([huge]),
            max_result_chars=40,
        ),
    )

    result = await loop.run("get huge chunk", request_id="req-7")

    assert result.answer.answer.startswith("Got a truncated")
    second_input = client.tool_turn_inputs[1]
    outputs = [
        item["output"]
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert outputs
    assert len(outputs[0]) <= 40


@pytest.mark.asyncio
async def test_max_total_calls_bound():
    """Fail closed when total tool executions exceed max_calls."""
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-1",
                        name="search_documents",
                        arguments={"query": "a"},
                    ),
                    ToolCallProposal(
                        call_id="call-2",
                        name="search_documents",
                        arguments={"query": "b"},
                    ),
                ],
            ),
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-3",
                        name="search_documents",
                        arguments={"query": "c"},
                    )
                ],
            ),
        ]
    )
    loop = _loop(client, max_turns=5, max_calls=2, max_calls_per_turn=5)

    with pytest.raises(StructuredOutputError) as exc_info:
        await loop.run("too many calls", request_id="req-8")

    assert exc_info.value.category == "unsupported"
