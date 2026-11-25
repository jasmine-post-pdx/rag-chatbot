"""Validate and execute allow-listed read-only document tools safely."""

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.domain.tools import (
    GetDocumentChunkArgs,
    SearchDocumentsArgs,
    ToolCallProposal,
    ToolObservation,
)
from app.retrieval.document_store import DocumentStore, SearchHit
from app.tools.registry import ALLOWED_TOOL_NAMES

logger = logging.getLogger(__name__)


def _sanitize_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a log-safe summary of tool arguments without document text.

    :param name: Tool name being executed.
    :param arguments: Raw model arguments.
    :returns: Sanitized argument metadata safe for logs."""
    keys = sorted(str(key) for key in arguments)
    summary: dict[str, Any] = {"arg_keys": keys}
    if name == "search_documents":
        query = arguments.get("query")
        if isinstance(query, str):
            summary["query_chars"] = len(query)
        limit = arguments.get("limit")
        if isinstance(limit, int):
            summary["limit"] = limit
        if "framework" in arguments:
            summary["has_framework"] = arguments.get("framework") is not None
        if "language" in arguments:
            summary["has_language"] = arguments.get("language") is not None
    elif name == "get_document_chunk":
        chunk_id = arguments.get("chunk_id")
        if isinstance(chunk_id, str):
            summary["chunk_id_chars"] = len(chunk_id)
    return summary


def _cap_content(content: str, max_chars: int) -> tuple[str, bool]:
    """Truncate content to the configured character cap.

    :param content: Full observation content.
    :param max_chars: Maximum characters to keep.
    :returns: Possibly truncated content and whether truncation occurred."""
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _serialize_search_hits(hits: list[SearchHit]) -> str:
    """Serialize search hits as compact JSON for the model.

    :param hits: Search results from the document store.
    :returns: JSON array string."""
    return json.dumps(
        [hit.model_dump() for hit in hits],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _serialize_chunk(chunk: object) -> str:
    """Serialize a document chunk as compact JSON for the model.

    :param chunk: Document chunk or ``None``.
    :returns: JSON object string, or a not-found payload."""
    if chunk is None:
        return json.dumps({"found": False}, separators=(",", ":"))
    payload = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
    payload = {**payload, "found": True}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class SafeToolExecutor:
    """Execute only allow-listed tools under deadline and size caps.

    Invalid names, invalid arguments, timeouts, and exceptions become safe
    error observations rather than uncaught failures. Document text is never
    written to logs.

    :ivar _store: Read-only document store used by allow-listed tools.
    :ivar _max_result_chars: Maximum observation content length.
    :ivar _call_timeout_seconds: Per-call deadline.
    :ivar _search_max_limit: Hard maximum for ``search_documents.limit``."""

    def __init__(
        self,
        store: DocumentStore,
        *,
        max_result_chars: int = 4000,
        call_timeout_seconds: float = 2.0,
        search_max_limit: int = 5,
    ) -> None:
        """Initialize the executor.

        :param store: Document store used for search and chunk lookup.
        :param max_result_chars: Observation content character cap.
        :param call_timeout_seconds: Deadline for each tool call.
        :param search_max_limit: Hard max for search ``limit`` arguments."""
        self._store = store
        self._max_result_chars = max_result_chars
        self._call_timeout_seconds = call_timeout_seconds
        self._search_max_limit = search_max_limit

    async def execute(
        self,
        proposal: ToolCallProposal,
        *,
        request_id: str | None = None,
    ) -> ToolObservation:
        """Validate and execute one proposed tool call.

        :param proposal: Model-proposed tool name and arguments.
        :param request_id: Optional HTTP request ID for log correlation.
        :returns: A success or safe-error observation for the model loop."""
        started = time.perf_counter()
        success = False
        error: str | None = None
        content = ""
        truncated = False

        try:
            if proposal.name not in ALLOWED_TOOL_NAMES:
                error = "unknown_tool"
                content = json.dumps(
                    {"error": "unknown_tool", "name": proposal.name},
                    separators=(",", ":"),
                )
            else:
                content, truncated = await self._run_allowed(proposal)
                success = True
        except ValidationError:
            error = "invalid_arguments"
            content = json.dumps(
                {"error": "invalid_arguments", "name": proposal.name},
                separators=(",", ":"),
            )
        except TimeoutError:
            error = "timeout"
            content = json.dumps(
                {"error": "timeout", "name": proposal.name},
                separators=(",", ":"),
            )
        except Exception:  # noqa: BLE001 — map any store failure to a safe observation
            error = "execution_error"
            content = json.dumps(
                {"error": "execution_error", "name": proposal.name},
                separators=(",", ":"),
            )

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "tool_call",
            extra={
                "request_id": request_id,
                "tool_name": proposal.name,
                "tool_args": _sanitize_arguments(proposal.name, proposal.arguments),
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
                "truncated": truncated,
            },
        )
        return ToolObservation(
            call_id=proposal.call_id,
            name=proposal.name,
            success=success,
            content=content,
            error=error,
            truncated=truncated,
        )

    async def _run_allowed(self, proposal: ToolCallProposal) -> tuple[str, bool]:
        """Validate arguments and run an allow-listed tool under a deadline.

        :param proposal: Allow-listed tool proposal.
        :returns: Serialized content and whether it was truncated.
        :raises ValidationError: When arguments fail local schema checks.
        :raises TimeoutError: When the call exceeds the configured deadline.
        :raises Exception: When the underlying store raises."""
        if proposal.name == "search_documents":
            args = SearchDocumentsArgs.model_validate(proposal.arguments)
            if args.limit > self._search_max_limit:
                raise ValidationError.from_exception_data(
                    "SearchDocumentsArgs",
                    [
                        {
                            "type": "less_than_equal",
                            "loc": ("limit",),
                            "input": args.limit,
                            "ctx": {"le": self._search_max_limit},
                        }
                    ],
                )

            async def _search() -> str:
                hits = await asyncio.to_thread(
                    self._store.search_documents,
                    args.query,
                    framework=args.framework,
                    language=args.language,
                    limit=args.limit,
                )
                return _serialize_search_hits(hits)

            raw = await asyncio.wait_for(
                _search(),
                timeout=self._call_timeout_seconds,
            )
            return _cap_content(raw, self._max_result_chars)

        if proposal.name == "get_document_chunk":
            args = GetDocumentChunkArgs.model_validate(proposal.arguments)

            async def _get() -> str:
                chunk = await asyncio.to_thread(
                    self._store.get_document_chunk,
                    args.chunk_id,
                )
                return _serialize_chunk(chunk)

            raw = await asyncio.wait_for(
                _get(),
                timeout=self._call_timeout_seconds,
            )
            return _cap_content(raw, self._max_result_chars)

        # Defensive: allow-list checked by caller.
        raise ValidationError.from_exception_data(
            "ToolCallProposal",
            [
                {
                    "type": "value_error",
                    "loc": ("name",),
                    "input": proposal.name,
                    "ctx": {"error": "unknown tool"},
                }
            ],
        )
