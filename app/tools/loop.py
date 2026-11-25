"""Bounded model → tool → observation → answer orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.domain.errors import StructuredOutputError
from app.domain.model_outputs import AnswerV1
from app.domain.tools import ToolObservation
from app.llm.generation import LLMClient, StructuredGenerationResult
from app.tools.executor import SafeToolExecutor
from app.tools.registry import tool_definitions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolLoopResult:
    """Outcome of a bounded tool loop.

    :ivar answer: Final locally validated ``AnswerV1``.
    :ivar structured: Provider metadata wrapper around the final answer.
    :ivar tool_calls_executed: Number of tool proposals executed.
    :ivar documents_considered: Distinct source documents seen in successful
        search/get observations (best-effort from observation JSON).
    :ivar chunks_returned: Distinct chunk IDs seen in successful observations."""

    answer: AnswerV1
    structured: StructuredGenerationResult[AnswerV1]
    tool_calls_executed: int
    documents_considered: int
    chunks_returned: int


class ToolLoop:
    """Run allow-listed tools under turn and call bounds.

    :ivar _client: LLM client that proposes tool calls or a final answer.
    :ivar _executor: Safe executor for allow-listed document tools.
    :ivar _max_turns: Maximum model↔tool iterations.
    :ivar _max_calls: Maximum total tool executions per question.
    :ivar _max_calls_per_turn: Maximum proposals accepted from one model turn.
    :ivar _search_max_limit: Hard max advertised in tool schemas."""

    def __init__(
        self,
        client: LLMClient,
        executor: SafeToolExecutor,
        *,
        max_turns: int = 3,
        max_calls: int = 6,
        max_calls_per_turn: int = 3,
        search_max_limit: int = 5,
    ) -> None:
        """Initialize the loop.

        :param client: LLM client implementing ``generate_tool_turn``.
        :param executor: Safe tool executor.
        :param max_turns: Maximum model turns including the final answer turn.
        :param max_calls: Maximum total tool executions.
        :param max_calls_per_turn: Cap on tool proposals from one turn.
        :param search_max_limit: Hard max for search tool schemas."""
        self._client = client
        self._executor = executor
        self._max_turns = max_turns
        self._max_calls = max_calls
        self._max_calls_per_turn = max_calls_per_turn
        self._search_max_limit = search_max_limit

    async def run(
        self,
        question: str,
        *,
        request_id: str | None = None,
        instructions: str | None = None,
    ) -> ToolLoopResult:
        """Execute the tool loop until a final answer or a bound is hit.

        :param question: User question text.
        :param request_id: Optional HTTP request ID for log correlation.
        :param instructions: Optional instructions prepended as a system message.
        :returns: Final answer plus retrieval/tool activity metrics.
        :raises StructuredOutputError: When turn/call bounds are exceeded or the
            model fails to produce a valid final answer."""
        tools = tool_definitions(search_max_limit=self._search_max_limit)
        input_items: list[dict[str, Any]] = []
        if instructions:
            input_items.append({"role": "system", "content": instructions})
        input_items.append({"role": "user", "content": question})

        tool_calls_executed = 0
        source_ids: set[str] = set()
        chunk_ids: set[str] = set()

        for turn_index in range(self._max_turns):
            turn_input: str | list[dict[str, Any]]
            if turn_index == 0 and len(input_items) == 1 and input_items[0]["role"] == "user":
                turn_input = question
            else:
                turn_input = list(input_items)

            turn = await self._client.generate_tool_turn(
                turn_input,
                tools=tools,
                output_model=AnswerV1,
            )

            if turn.kind == "structured":
                if turn.structured is None:
                    raise StructuredOutputError("validation")
                return ToolLoopResult(
                    answer=turn.structured.parsed,
                    structured=turn.structured,
                    tool_calls_executed=tool_calls_executed,
                    documents_considered=len(source_ids),
                    chunks_returned=len(chunk_ids),
                )

            proposals = list(turn.tool_calls or [])
            if not proposals:
                raise StructuredOutputError(
                    "validation",
                    "Tool turn returned neither tool calls nor structured output",
                )

            # Echo function_call items so the provider sees the prior proposals.
            for proposal in proposals[: self._max_calls_per_turn]:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": proposal.call_id,
                        "name": proposal.name,
                        "arguments": json.dumps(proposal.arguments, separators=(",", ":")),
                    }
                )

            accepted = proposals[: self._max_calls_per_turn]
            for proposal in accepted:
                if tool_calls_executed >= self._max_calls:
                    raise StructuredOutputError(
                        "unsupported",
                        "Tool call bound exceeded",
                    )
                observation = await self._executor.execute(
                    proposal,
                    request_id=request_id,
                )
                tool_calls_executed += 1
                _accumulate_ids(observation, source_ids, chunk_ids)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": observation.call_id,
                        "output": observation.content,
                    }
                )
                logger.info(
                    "tool_loop_turn",
                    extra={
                        "request_id": request_id,
                        "turn_index": turn_index,
                        "tool_name": proposal.name,
                        "success": observation.success,
                        "error": observation.error,
                    },
                )

        raise StructuredOutputError(
            "unsupported",
            "Tool turn bound exceeded without a final answer",
        )


def _accumulate_ids(
    observation: ToolObservation,
    source_ids: set[str],
    chunk_ids: set[str],
) -> None:
    """Best-effort collect source/chunk IDs from successful observation JSON.

    :param observation: Tool observation returned to the model.
    :param source_ids: Mutable set of source identifiers.
    :param chunk_ids: Mutable set of chunk identifiers."""
    if not observation.success or not observation.content:
        return
    try:
        payload = json.loads(observation.content)
    except json.JSONDecodeError:
        return
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            chunk_id = item.get("chunk_id")
            if isinstance(source_id, str) and source_id:
                source_ids.add(source_id)
            if isinstance(chunk_id, str) and chunk_id:
                chunk_ids.add(chunk_id)
    elif isinstance(payload, dict):
        if payload.get("found") is False:
            return
        source_id = payload.get("source_id")
        chunk_id = payload.get("chunk_id")
        if isinstance(source_id, str) and source_id:
            source_ids.add(source_id)
        if isinstance(chunk_id, str) and chunk_id:
            chunk_ids.add(chunk_id)
