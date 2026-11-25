"""Unit tests for allow-listed read-only tool execution."""

import logging
import time

import pytest

from app.domain.tools import ToolCallProposal, ToolObservation
from app.retrieval.document_store import DocumentChunk
from app.retrieval.memory import InMemoryDocumentStore
from app.tools.executor import SafeToolExecutor
from app.tools.registry import ALLOWED_TOOL_NAMES, tool_definitions, tool_schema_strings


def _store() -> InMemoryDocumentStore:
    """Build a small store for executor tests.

    :returns: An in-memory document store with two chunks."""
    return InMemoryDocumentStore(
        [
            DocumentChunk(
                source_id="fastapi-guide",
                chunk_id="c1",
                text="FastAPI dependency injection wires routers to services.",
                framework="fastapi",
                language="python",
            ),
            DocumentChunk(
                source_id="rag-notes",
                chunk_id="c2",
                text="Context budgeting reserves tokens for tool schemas.",
            ),
        ]
    )


def _executor(**kwargs) -> SafeToolExecutor:
    """Build a safe executor over the test store.

    :param kwargs: Overrides for executor configuration.
    :returns: A ``SafeToolExecutor`` instance."""
    defaults = {
        "store": _store(),
        "max_result_chars": 4000,
        "call_timeout_seconds": 2.0,
        "search_max_limit": 5,
    }
    defaults.update(kwargs)
    return SafeToolExecutor(**defaults)


@pytest.mark.asyncio
async def test_search_documents_returns_success_observation():
    """Execute a normal search_documents call successfully."""
    executor = _executor()
    proposal = ToolCallProposal(
        call_id="call-1",
        name="search_documents",
        arguments={"query": "dependency injection", "limit": 5},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is True
    assert observation.name == "search_documents"
    assert observation.call_id == "call-1"
    assert observation.error is None
    assert "c1" in observation.content
    assert "dependency injection" in observation.content.lower()


@pytest.mark.asyncio
async def test_get_document_chunk_returns_success_observation():
    """Execute a normal get_document_chunk call successfully."""
    executor = _executor()
    proposal = ToolCallProposal(
        call_id="call-2",
        name="get_document_chunk",
        arguments={"chunk_id": "c1"},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is True
    assert observation.name == "get_document_chunk"
    assert "c1" in observation.content
    assert "FastAPI" in observation.content


@pytest.mark.asyncio
async def test_invented_tool_name_returns_safe_error_and_never_executes():
    """Deny invented tool names with a safe error observation."""
    class RecordingStore(InMemoryDocumentStore):
        def __init__(self) -> None:
            super().__init__([])
            self.searches = 0
            self.gets = 0

        def search_documents(self, *args, **kwargs):
            self.searches += 1
            return super().search_documents(*args, **kwargs)

        def get_document_chunk(self, chunk_id: str):
            self.gets += 1
            return super().get_document_chunk(chunk_id)

    store = RecordingStore()
    executor = _executor(store=store)
    proposal = ToolCallProposal(
        call_id="call-x",
        name="admin_shell",
        arguments={"command": "rm -rf /"},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is False
    assert observation.error == "unknown_tool"
    assert store.searches == 0
    assert store.gets == 0


@pytest.mark.asyncio
async def test_injection_like_hidden_admin_tool_is_denied():
    """Reject injection-like requests for a hidden admin tool via allow-list."""
    executor = _executor()
    proposal = ToolCallProposal(
        call_id="call-inj",
        name="run_sql",
        arguments={"sql": "DROP TABLE users"},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is False
    assert observation.error == "unknown_tool"
    assert "DROP" not in (observation.content or "")


@pytest.mark.asyncio
async def test_invalid_arguments_return_safe_error():
    """Reject missing or invalid tool arguments locally."""
    executor = _executor()
    proposal = ToolCallProposal(
        call_id="call-bad",
        name="search_documents",
        arguments={},  # missing required query
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is False
    assert observation.error == "invalid_arguments"


@pytest.mark.asyncio
async def test_excessive_limit_is_rejected():
    """Reject a search limit above the hard maximum."""
    executor = _executor(search_max_limit=5)
    proposal = ToolCallProposal(
        call_id="call-limit",
        name="search_documents",
        arguments={"query": "fastapi", "limit": 100},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is False
    assert observation.error == "invalid_arguments"


@pytest.mark.asyncio
async def test_handler_exception_returns_safe_error_observation():
    """Map handler exceptions to a safe error observation."""

    class BoomStore:
        def search_documents(self, *args, **kwargs):
            raise RuntimeError("secret internal failure with password=hunter2")

        def get_document_chunk(self, chunk_id: str):
            raise RuntimeError("secret")

    executor = _executor(store=BoomStore())
    proposal = ToolCallProposal(
        call_id="call-boom",
        name="search_documents",
        arguments={"query": "x"},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is False
    assert observation.error == "execution_error"
    assert "hunter2" not in observation.content
    assert "password" not in observation.content.lower()


@pytest.mark.asyncio
async def test_handler_timeout_returns_safe_error_observation():
    """Map a handler deadline miss to a safe timeout observation."""

    class SlowStore:
        def search_documents(self, *args, **kwargs):
            time.sleep(0.2)
            return []

        def get_document_chunk(self, chunk_id: str):
            return None

    executor = _executor(store=SlowStore(), call_timeout_seconds=0.05)
    proposal = ToolCallProposal(
        call_id="call-slow",
        name="search_documents",
        arguments={"query": "x"},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is False
    assert observation.error == "timeout"


@pytest.mark.asyncio
async def test_output_size_over_cap_is_truncated():
    """Truncate oversized tool content to the configured character cap."""
    huge = DocumentChunk(
        source_id="big",
        chunk_id="huge",
        text="A" * 500,
    )
    executor = _executor(
        store=InMemoryDocumentStore([huge]),
        max_result_chars=50,
    )
    proposal = ToolCallProposal(
        call_id="call-cap",
        name="get_document_chunk",
        arguments={"chunk_id": "huge"},
    )

    observation = await executor.execute(proposal, request_id="req-1")

    assert observation.success is True
    assert len(observation.content) <= 50
    assert observation.truncated is True


@pytest.mark.asyncio
async def test_executor_logs_safe_fields_without_document_text(caplog):
    """Log request ID, tool name, sanitized args, duration, and success."""
    executor = _executor()
    proposal = ToolCallProposal(
        call_id="call-log",
        name="search_documents",
        arguments={"query": "dependency injection", "limit": 2},
    )

    with caplog.at_level(logging.INFO, logger="app.tools.executor"):
        await executor.execute(proposal, request_id="req-log")

    record = next(
        item
        for item in caplog.records
        if item.name == "app.tools.executor" and item.message == "tool_call"
    )
    assert record.request_id == "req-log"
    assert record.tool_name == "search_documents"
    assert record.success is True
    assert record.duration_ms >= 0
    assert record.tool_args["arg_keys"] == ["limit", "query"]
    assert record.tool_args["query_chars"] == len("dependency injection")
    joined = f"{record.getMessage()} {record.__dict__}"
    assert "dependency injection" not in joined
    assert "wires routers" not in joined


def test_registry_allow_list_and_schema_strings():
    """Expose only the two read-only tools with serializable schemas."""
    assert ALLOWED_TOOL_NAMES == frozenset(
        {"search_documents", "get_document_chunk"}
    )
    definitions = tool_definitions(search_max_limit=5)
    assert {item["name"] for item in definitions} == ALLOWED_TOOL_NAMES
    schemas = tool_schema_strings(search_max_limit=5)
    assert len(schemas) == 2
    assert all(isinstance(item, str) for item in schemas)
    assert any("search_documents" in item for item in schemas)
    assert any("get_document_chunk" in item for item in schemas)


def test_tool_observation_model_defaults():
    """ToolObservation carries success/error fields for the model loop."""
    obs = ToolObservation(
        call_id="c",
        name="search_documents",
        success=True,
        content="[]",
    )
    assert obs.error is None
    assert obs.truncated is False
