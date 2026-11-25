"""Provider-neutral interfaces and models for text generation."""

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from app.domain.errors import StructuredOutputError
from app.domain.tools import ToolCallProposal

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

# Rate limits and selected server errors. Other 4xx responses are not retried.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0
# Timeouts/connection drops are ambiguous: the provider may already have billed.
_TRANSPORT_MAX_RETRIES = 1
_MAX_RETRY_WALL_SECONDS = 60.0

AsyncSleeper = Callable[[float], Awaitable[None]]


async def _asyncio_sleep(seconds: float) -> None:
    """Sleep without blocking the event loop.

    :param seconds: Seconds to wait before returning."""
    await asyncio.sleep(seconds)


class GenerationResult(BaseModel):
    """Generated answer and the metadata the provider actually reported.

    :ivar answer: Text produced by the model, or the refusal text when the model
        returned no output text.
    :ivar provider: Provider identifier, such as ``"openai"`` or ``"local"``.
    :ivar model: Model identifier reported for the generation.
    :ivar input_tokens: Input tokens reported by the provider, or ``None`` when
        the provider omitted usage.
    :ivar output_tokens: Output tokens reported by the provider, or ``None`` when
        the provider omitted usage.
    :ivar finish_status: Provider finish status, such as ``"completed"``.
    :ivar refusal: Refusal text when the provider returned one.
    :ivar provider_request_id: Provider response ID for this generation."""

    answer: str
    provider: str
    model: str
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    finish_status: str | None = None
    refusal: str | None = None
    provider_request_id: str | None = None


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[TModel]):
    """Schema-constrained generation outcome with provider metadata.

    :ivar parsed: Locally validated structured output.
    :ivar provider: Provider identifier, such as ``"openai"`` or ``"fake"``.
    :ivar model: Model identifier reported for the generation.
    :ivar input_tokens: Input tokens reported by the provider, or ``None``.
    :ivar output_tokens: Output tokens reported by the provider, or ``None``.
    :ivar finish_status: Provider finish status, such as ``"completed"``.
    :ivar refusal: Refusal text when the provider returned one.
    :ivar provider_request_id: Provider response ID for this generation."""

    parsed: TModel
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_status: str | None = None
    refusal: str | None = None
    provider_request_id: str | None = None


@dataclass(frozen=True)
class ToolTurnResult(Generic[TModel]):
    """One model turn that either proposes tool calls or returns structured output.

    :ivar kind: ``"tool_calls"`` when the model requested tools, else ``"structured"``.
    :ivar tool_calls: Proposed tool calls when ``kind`` is ``"tool_calls"``.
    :ivar structured: Final structured result when ``kind`` is ``"structured"``."""

    kind: Literal["tool_calls", "structured"]
    tool_calls: list[ToolCallProposal] | None = None
    structured: StructuredGenerationResult[TModel] | None = None


class LLMClient(Protocol):
    """Interface implemented by language model generation clients.

    Implementations return provider-reported token usage with each answer.
    Callers must await ``generate``, ``generate_structured``, and
    ``generate_tool_turn``.
    """

    async def generate(self, question: str) -> GenerationResult:
        """Generate an answer for a question.

        :param question: User question to answer.
        :returns: Generated text with provider, model, and token usage metadata."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        output_model: type[TModel],
    ) -> StructuredGenerationResult[TModel]:
        """Generate a schema-constrained payload and validate it locally.

        :param prompt: Prompt text sent as generation input.
        :param output_model: Pydantic model describing the expected output.
        :returns: Parsed model instance with provider metadata.
        :raises StructuredOutputError: On refusal, incomplete, or invalid output."""
        ...

    async def generate_tool_turn(
        self,
        input_items: str | Sequence[object],
        tools: Sequence[dict[str, Any]],
        output_model: type[TModel],
    ) -> ToolTurnResult[TModel]:
        """Run one tool-enabled turn returning tool calls or structured output.

        :param input_items: Prompt string or multi-turn Responses input items.
        :param tools: Allow-listed tool definitions sent to the provider.
        :param output_model: Pydantic model for a final structured answer.
        :returns: Either proposed tool calls or a validated structured result.
        :raises StructuredOutputError: On refusal, incomplete, or invalid output."""
        ...


class FakeLLMClient:
    """Return deterministic responses without contacting a model provider.

    :ivar model: Model identifier reported in each generated result.
    :ivar _structured_payload: Optional object or mapping returned by
        ``generate_structured`` after local validation.
    :ivar _structured_refusal: When set, structured generation fails as refusal.
    :ivar _structured_incomplete: When true, structured generation fails as incomplete.
    :ivar _structured_finish_status: Finish status reported on structured success.
    :ivar _tool_turns: Scripted sequence consumed by ``generate_tool_turn``."""

    def __init__(
        self,
        model: str = "fake",
        *,
        structured_payload: BaseModel | dict | None = None,
        structured_refusal: str | None = None,
        structured_incomplete: bool = False,
        structured_finish_status: str | None = "completed",
        tool_turns: list[ToolTurnResult] | None = None,
    ) -> None:
        """Initialize the fake client.

        :param model: Model identifier to report in generated results.
        :param structured_payload: Canned structured output for
            ``generate_structured``. Validated against the requested model.
        :param structured_refusal: When set, ``generate_structured`` raises a
            refusal ``StructuredOutputError``.
        :param structured_incomplete: When true, ``generate_structured`` raises
            an incomplete ``StructuredOutputError``.
        :param structured_finish_status: Finish status on structured success.
        :param tool_turns: Optional scripted tool-turn sequence. Each
            ``generate_tool_turn`` call consumes the next entry."""
        self.model = model
        self.provider = "fake"
        self._structured_payload = structured_payload
        self._structured_refusal = structured_refusal
        self._structured_incomplete = structured_incomplete
        self._structured_finish_status = structured_finish_status
        self._tool_turns = list(tool_turns) if tool_turns is not None else None
        self.tool_turn_inputs: list[object] = []

    async def generate(self, question: str) -> GenerationResult:
        """Return a stable answer without making a provider request.

        :param question: User question to answer.
        :returns: A deterministic answer with zero token usage."""
        return GenerationResult(
            answer=f"Fake answer to: {question}",
            provider="fake",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
        )

    async def generate_structured(
        self,
        prompt: str,
        output_model: type[TModel],
    ) -> StructuredGenerationResult[TModel]:
        """Return a canned structured payload or a controlled structured failure.

        :param prompt: Prompt text; unused except to satisfy the interface.
        :param output_model: Pydantic model used to validate the canned payload.
        :returns: Parsed model with zero token usage.
        :raises StructuredOutputError: On configured refusal, incomplete output,
            missing payload, or local validation failure."""
        if self._structured_refusal is not None:
            raise StructuredOutputError("refusal")
        if self._structured_incomplete:
            raise StructuredOutputError("incomplete")
        try:
            if self._structured_payload is None:
                # Default AnswerV1-shaped abstention so unconfigured fakes still work.
                parsed = output_model.model_validate(
                    {
                        "answer": f"Fake answer to: {prompt}",
                        "claims": [],
                        "insufficient_evidence": True,
                    }
                )
            elif isinstance(self._structured_payload, output_model):
                parsed = self._structured_payload
            elif isinstance(self._structured_payload, BaseModel):
                parsed = output_model.model_validate(
                    self._structured_payload.model_dump()
                )
            else:
                parsed = output_model.model_validate(self._structured_payload)
        except ValidationError as exc:
            if self._structured_payload is None:
                raise StructuredOutputError(
                    "unsupported",
                    "FakeLLMClient has no structured_payload configured for "
                    f"{output_model.__name__}",
                ) from exc
            raise StructuredOutputError("validation") from exc
        return StructuredGenerationResult(
            parsed=parsed,
            provider="fake",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            finish_status=self._structured_finish_status,
        )

    async def generate_tool_turn(
        self,
        input_items: str | Sequence[object],
        tools: Sequence[dict[str, Any]],
        output_model: type[TModel],
    ) -> ToolTurnResult[TModel]:
        """Return the next scripted tool turn, or fail when the script is empty.

        :param input_items: Prompt or conversation items; recorded for assertions.
        :param tools: Tool definitions; unused except to satisfy the interface.
        :param output_model: Expected final structured model type.
        :returns: The next scripted ``ToolTurnResult``.
        :raises StructuredOutputError: When no scripted turns remain."""
        del tools  # Interface parity with real clients.
        self.tool_turn_inputs.append(input_items)
        if self._tool_turns is None:
            structured = await self.generate_structured(
                input_items if isinstance(input_items, str) else str(input_items),
                output_model,
            )
            return ToolTurnResult(kind="structured", structured=structured)
        if not self._tool_turns:
            raise StructuredOutputError(
                "unsupported",
                "FakeLLMClient has no remaining scripted tool turns",
            )
        return self._tool_turns.pop(0)


class MissingAPIKeyError(RuntimeError):
    """A provider client was used without its required API key configured."""


class OpenAIResponsesClient:
    """Generate text with OpenAI's async Python SDK and Responses API.

    Prefer injecting a shared ``AsyncOpenAI`` client owned by the FastAPI
    lifespan. When no client is injected, one is created on first use so the
    application can start without ``OPENAI_API_KEY`` and only fails when
    generation is attempted. Lifespan-owned clients are closed on shutdown;
    lazily created clients are not closed by this adapter.

    SDK retries are disabled. This client retries rate limits and selected 5xx
    responses up to ``max_retries``. Timeout and connection failures are retried
    at most once, and no retry starts after the wall-clock budget is exhausted.
    Authentication failures and bad requests are not retried.
    ``responses.create`` is not idempotent, so a transport retry can bill a
    second generation when an earlier request reached the provider; those
    retries attach ``possible_duplicate_generation`` to the warning log
    record (with ``retry_class`` and ``elapsed_seconds``).

    Per-attempt timeouts apply to each Responses API call. Bounded retries and
    backoff can make total generation latency longer than one
    ``timeout_seconds`` value.

    :ivar _client: Async SDK client used to make Responses API requests, if created.
    :ivar _model: Model identifier passed to each request.
    :ivar _timeout_seconds: Per-request timeout used when creating the SDK client.
    :ivar _max_output_tokens: Optional completion-token cap sent on each request.
    :ivar _instructions: Optional instructions sent separately from the question.
    :ivar _max_retries: Extra attempts after the first request.
    :ivar _sleeper: Awaitable wait used between retries.
    :ivar _rng: Returns a jitter factor in the range 0 to 1.
    :ivar _clock: Monotonic clock used for the retry wall-clock budget."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client=None,
        *,
        timeout_seconds: float = 30.0,
        max_output_tokens: int | None = None,
        instructions: str | None = None,
        max_retries: int = 2,
        sleeper: AsyncSleeper = _asyncio_sleep,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the adapter with a model and optional async SDK client.

        :param model: OpenAI model identifier to use for requests.
        :param api_key: Optional key from application settings.
        :param client: Optional async OpenAI SDK client, useful for isolated tests.
        :param timeout_seconds: Per-request timeout in seconds for a client this
            adapter creates. Ignored when ``client`` is injected.
        :param max_output_tokens: Optional completion-token cap. Omitted from the
            request when ``None``.
        :param instructions: Optional instructions sent separately from the
            question. Omitted from the request when ``None``.
        :param max_retries: Extra attempts after the first request. Zero disables
            application retries. Transport failures still cap at one retry.
        :param sleeper: Awaitable wait used between retries. Tests can replace it.
        :param rng: Jitter source used for backoff. Tests can replace it.
        :param clock: Monotonic clock used for the retry wall-clock budget.
        :raises ValueError: If ``max_retries`` is negative."""
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        self._client = client
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._instructions = instructions
        self._max_retries = max_retries
        self._sleeper = sleeper
        self._rng = rng
        self._clock = clock
        self.provider = "openai"
        self.model = model

    def _get_client(self):
        """Return the async SDK client, creating it on first use.

        :returns: The async OpenAI SDK client.
        :raises MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set."""
        if self._client is None:
            if not self._api_key:
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set; it is required when "
                    "GENERATION_PROVIDER is 'openai'"
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout_seconds,
                max_retries=0,
            )
        return self._client

    def _request_body(self, question: str | Sequence[object]) -> dict[str, object]:
        """Build the Responses API request from the question and client settings.

        :param question: User question or multi-turn input items for the request.
        :returns: Request fields. Instructions and the output cap are included only
            when configured."""
        body: dict[str, object] = {"model": self._model, "input": question}
        if self._instructions is not None:
            body["instructions"] = self._instructions
        if self._max_output_tokens is not None:
            body["max_output_tokens"] = self._max_output_tokens
        return body

    async def _with_retries(self, call: Callable[[], Awaitable[object]]) -> object:
        """Await a provider call with the shared retry policy.

        :param call: Zero-argument async callable that performs one attempt.
        :returns: The successful provider response object.
        :raises Exception: The provider error from the final attempt."""
        started = self._clock()
        transport_retries = 0
        for attempt in range(self._max_retries + 1):
            try:
                return await call()
            except Exception as exc:
                kind = _retry_class(exc)
                elapsed = self._clock() - started
                within_budget = elapsed < _MAX_RETRY_WALL_SECONDS
                within_max = attempt < self._max_retries
                if kind == "transport":
                    class_ok = transport_retries < _TRANSPORT_MAX_RETRIES
                    possible_duplicate = True
                elif kind == "status":
                    class_ok = True
                    possible_duplicate = False
                else:
                    class_ok = False
                    possible_duplicate = False
                will_retry = (
                    kind is not None and within_max and within_budget and class_ok
                )
                if not will_retry:
                    raise
                if kind == "transport":
                    transport_retries += 1
                logger.warning(
                    "Retrying generation",
                    extra={
                        "provider": self.provider,
                        "model": self.model,
                        "attempt": attempt + 1,
                        "provider_status": getattr(exc, "status_code", None),
                        "exception_type": type(exc).__name__,
                        "retry_class": kind,
                        "possible_duplicate_generation": possible_duplicate,
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )
                await self._sleeper(_retry_delay_seconds(attempt, self._rng()))
        raise RuntimeError("generation retry loop ended without a result")

    async def generate(self, question: str) -> GenerationResult:
        """Generate an answer and return provider-reported token usage.

        :param question: User question to send as the Responses API input.
        :returns: Generated text, the model identifier, and reported token usage.
        :raises MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set.
        :raises Exception: The provider error from the final attempt. Retryable
            failures are retried up to ``max_retries`` extra times."""
        client = self._get_client()
        body = self._request_body(question)

        async def _create() -> object:
            return await client.responses.create(**body)

        response = await self._with_retries(_create)
        return _generation_result(response, self._model)

    async def generate_structured(
        self,
        prompt: str,
        output_model: type[TModel],
    ) -> StructuredGenerationResult[TModel]:
        """Generate schema-constrained output via Responses ``parse``, then re-validate.

        Uses the provider's structured-output support when available, then validates
        the parsed payload locally against ``output_model`` (including invariants).
        Refusals, incomplete finishes, missing parsed payloads, and local validation
        failures become ``StructuredOutputError`` without inventing fields.

        :param prompt: Prompt text sent as the Responses API input.
        :param output_model: Pydantic model describing the expected output.
        :returns: Locally validated parsed model with provider metadata.
        :raises MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set.
        :raises StructuredOutputError: On refusal, incomplete, unsupported, or
            invalid structured output.
        :raises Exception: Retryable provider transport/status errors after retries."""
        client = self._get_client()
        body = self._request_body(prompt)
        parse = getattr(client.responses, "parse", None)
        if not callable(parse):
            raise StructuredOutputError(
                "unsupported",
                "Provider client does not support schema-constrained responses.parse",
            )

        async def _parse() -> object:
            return await parse(**body, text_format=output_model)

        response = await self._with_retries(_parse)
        return _structured_generation_result(response, self._model, output_model)

    async def generate_tool_turn(
        self,
        input_items: str | Sequence[object],
        tools: Sequence[dict[str, Any]],
        output_model: type[TModel],
    ) -> ToolTurnResult[TModel]:
        """Run one Responses turn with tools, returning calls or structured output.

        Prefers ``responses.parse`` with ``text_format`` so a final answer can be
        schema-constrained. When the provider returns ``function_call`` items,
        those are mapped to ``ToolCallProposal`` values instead.

        :param input_items: Prompt string or multi-turn Responses input items.
        :param tools: Allow-listed tool definitions included in the request.
        :param output_model: Pydantic model for a final structured answer.
        :returns: Tool-call proposals or a locally validated structured result.
        :raises MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set.
        :raises StructuredOutputError: On refusal, incomplete, malformed tool
            arguments, or invalid structured output.
        :raises Exception: Retryable provider errors after retries are exhausted."""
        client = self._get_client()
        body = self._request_body(input_items)
        body["tools"] = list(tools)
        parse = getattr(client.responses, "parse", None)
        if not callable(parse):
            raise StructuredOutputError(
                "unsupported",
                "Provider client does not support schema-constrained responses.parse",
            )

        async def _parse() -> object:
            return await parse(**body, text_format=output_model)

        response = await self._with_retries(_parse)
        proposals = _tool_call_proposals(response)
        if proposals:
            return ToolTurnResult(kind="tool_calls", tool_calls=proposals)
        structured = _structured_generation_result(response, self._model, output_model)
        return ToolTurnResult(kind="structured", structured=structured)


def _tool_call_proposals(response: object) -> list[ToolCallProposal]:
    """Extract function_call items from a Responses API output list.

    :param response: Provider response object or mapping.
    :returns: Proposed tool calls in output order.
    :raises StructuredOutputError: When a function_call has malformed arguments."""
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return []

    proposals: list[ToolCallProposal] = []
    for item in output:
        item_type = _mapping_or_attr(item, "type")
        if item_type != "function_call":
            continue
        call_id = _mapping_or_attr(item, "call_id")
        name = _mapping_or_attr(item, "name")
        raw_arguments = _mapping_or_attr(item, "arguments")
        if not isinstance(call_id, str) or not call_id:
            raise StructuredOutputError("validation", "Tool call missing call_id")
        if not isinstance(name, str) or not name:
            raise StructuredOutputError("validation", "Tool call missing name")
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str):
            try:
                parsed_args = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise StructuredOutputError("validation") from exc
            if not isinstance(parsed_args, dict):
                raise StructuredOutputError("validation")
            arguments = parsed_args
        else:
            raise StructuredOutputError("validation")
        proposals.append(
            ToolCallProposal(call_id=call_id, name=name, arguments=arguments)
        )
    return proposals


def _generation_result(response: object, fallback_model: str) -> GenerationResult:
    """Map a Responses API response without inventing omitted metadata.

    :param response: Provider response object or mapping.
    :param fallback_model: Model identifier used when the response omits ``model``.
    :returns: A generation result with only the metadata the provider reported."""
    refusal = _refusal_text(response)
    reported_model = getattr(response, "model", None)
    if not isinstance(reported_model, str) or reported_model == "":
        reported_model = fallback_model
    return GenerationResult(
        answer=_answer_text(response, refusal),
        provider="openai",
        model=reported_model,
        input_tokens=_reported_token_count(getattr(response, "usage", None), "input_tokens"),
        output_tokens=_reported_token_count(getattr(response, "usage", None), "output_tokens"),
        finish_status=_finish_status(response),
        refusal=refusal,
        provider_request_id=_provider_request_id(response),
    )


def _structured_generation_result(
    response: object,
    fallback_model: str,
    output_model: type[TModel],
) -> StructuredGenerationResult[TModel]:
    """Map a parsed Responses API result and re-validate the structured payload.

    :param response: Provider parsed response object or mapping.
    :param fallback_model: Model identifier used when the response omits ``model``.
    :param output_model: Pydantic model used for local validation.
    :returns: Structured result with only provider-reported metadata.
    :raises StructuredOutputError: On refusal, incomplete finish, missing parsed
        payload, or local validation failure."""
    refusal = _refusal_text(response)
    finish_status = _finish_status(response)
    reported_model = getattr(response, "model", None)
    if not isinstance(reported_model, str) or reported_model == "":
        reported_model = fallback_model
    metadata = {
        "provider": "openai",
        "model": reported_model,
        "input_tokens": _reported_token_count(
            getattr(response, "usage", None), "input_tokens"
        ),
        "output_tokens": _reported_token_count(
            getattr(response, "usage", None), "output_tokens"
        ),
        "finish_status": finish_status,
        "refusal": refusal,
        "provider_request_id": _provider_request_id(response),
    }
    if refusal is not None:
        raise StructuredOutputError("refusal")
    if finish_status == "incomplete":
        raise StructuredOutputError("incomplete")
    parsed_value = getattr(response, "output_parsed", None)
    if parsed_value is None and isinstance(response, dict):
        parsed_value = response.get("output_parsed")
    if parsed_value is None:
        raise StructuredOutputError("validation")
    try:
        if isinstance(parsed_value, output_model):
            # Re-validate so domain invariants still run after SDK parsing.
            parsed = output_model.model_validate(parsed_value.model_dump())
        elif isinstance(parsed_value, BaseModel):
            parsed = output_model.model_validate(parsed_value.model_dump())
        else:
            parsed = output_model.model_validate(parsed_value)
    except ValidationError as exc:
        raise StructuredOutputError("validation") from exc
    return StructuredGenerationResult(parsed=parsed, **metadata)


def _retry_class(exc: Exception) -> str | None:
    """Classify a failure for the narrowed retry policy.

    :param exc: Error raised by one generation attempt.
    :returns: ``"status"`` for retryable HTTP statuses, ``"transport"`` for timeouts
        and connection failures, or ``None`` when the error must not be retried."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transport"
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
        return "status"
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return "transport"
    return None


def _is_retryable(exc: Exception) -> bool:
    """Return whether this failure may be retried under the policy.

    :param exc: Error raised by one generation attempt.
    :returns: ``True`` when the narrowed retry policy classifies the error as retryable."""
    return _retry_class(exc) is not None


def _retry_delay_seconds(failed_attempt: int, jitter: float) -> float:
    """Return an exponential backoff delay with equal jitter.

    :param failed_attempt: Zero for the delay after the first failure.
    :param jitter: Factor from 0 through 1 selecting a point between half the
        ceiling and the full ceiling.
    :returns: Seconds to wait before the next attempt. The ceiling doubles from
        half a second and stops growing at eight seconds. The delay is always
        in ``[ceiling / 2, ceiling]``."""
    bounded_jitter = min(1.0, max(0.0, jitter))
    ceiling = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2**failed_attempt))
    return ceiling * (0.5 + 0.5 * bounded_jitter)


def _reported_token_count(usage: object | None, name: str) -> int | None:
    """Return one usage count only when the provider reported it.

    :param usage: Provider usage object, or ``None`` when usage was omitted.
    :param name: Usage attribute to read, such as ``"input_tokens"``.
    :returns: The reported count, including zero. ``None`` when usage or the count
        was omitted."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    if value is None:
        return None
    return int(value)


def _finish_status(response: object) -> str | None:
    """Return the provider finish status string, when the response has one.

    :param response: Provider response object or mapping.
    :returns: The finish status string, or ``None`` when omitted."""
    status = getattr(response, "status", None)
    if status is None and isinstance(response, dict):
        status = response.get("status")
    if status is None:
        return None
    if isinstance(status, str):
        return status
    value = getattr(status, "value", None)
    if isinstance(value, str):
        return value
    return str(status)


def _provider_request_id(response: object) -> str | None:
    """Return the provider response ID, when the response has one.

    :param response: Provider response object or mapping.
    :returns: The provider response ID string, or ``None`` when omitted."""
    request_id = getattr(response, "id", None)
    if request_id is None and isinstance(response, dict):
        request_id = response.get("id")
    if request_id is None:
        return None
    return str(request_id)


def _output_content_texts(
    response: object,
    *,
    part_type: str,
    text_field: str,
) -> list[str]:
    """Collect non-empty text fields from ``output[]`` content parts.

    :param response: Provider response object or mapping.
    :param part_type: Content part type to include, such as ``"output_text"`` or
        ``"refusal"``.
    :param text_field: Field on matching parts that holds the text.
    :returns: Non-empty text values in output order."""
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if output is None:
        return []
    texts: list[str] = []
    for item in output:
        content = _mapping_or_attr(item, "content") or []
        for part in content:
            if _mapping_or_attr(part, "type") != part_type:
                continue
            text = _mapping_or_attr(part, text_field)
            if isinstance(text, str) and text:
                texts.append(text)
    return texts


def _refusal_text(response: object) -> str | None:
    """Collect refusal text from Responses API output content.

    :param response: Provider response object or mapping.
    :returns: Joined refusal text, or ``None`` when no refusal parts are present."""
    refusals = _output_content_texts(response, part_type="refusal", text_field="refusal")
    if not refusals:
        return None
    return "\n".join(refusals)


def _answer_text(response: object, refusal: str | None) -> str:
    """Return output text, or the refusal when the provider sent no output text.

    Prefers the SDK ``output_text`` helper when present and non-empty. Otherwise
    gathers ``output_text`` content parts from ``output[]``, then refusal text.

    :param response: Provider response object or mapping.
    :param refusal: Refusal text already extracted from the response, if any.
    :returns: The answer text to expose to callers."""
    output_text = getattr(response, "output_text", None)
    if output_text is None and isinstance(response, dict):
        output_text = response.get("output_text")
    structured_texts = _output_content_texts(
        response, part_type="output_text", text_field="text"
    )
    if isinstance(output_text, str) and output_text != "":
        return output_text
    if structured_texts:
        return "\n".join(structured_texts)
    if refusal is not None:
        return refusal
    if isinstance(output_text, str):
        return output_text
    return ""


def _mapping_or_attr(value: object, name: str) -> object:
    """Read a field from a mapping or an object.

    :param value: Mapping or object that may contain ``name``.
    :param name: Field name to read.
    :returns: The field value, or ``None`` when absent."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
