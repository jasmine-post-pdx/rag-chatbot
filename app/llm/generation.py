"""Provider-neutral interfaces and models for text generation."""

import logging
import random
import time
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Rate limits and selected server errors. Other 4xx responses are not retried.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0
# Timeouts/connection drops are ambiguous: the provider may already have billed.
_TRANSPORT_MAX_RETRIES = 1
_MAX_RETRY_WALL_SECONDS = 60.0


class GenerationResult(BaseModel):
    """Generated answer and the metadata the provider actually reported.

    Attributes:
        answer: Text produced by the model, or the refusal text when the model
            returned no output text.
        provider: Provider identifier, such as ``"openai"`` or ``"local"``.
        model: Model identifier reported for the generation.
        input_tokens: Input tokens reported by the provider, or ``None`` when
            the provider omitted usage.
        output_tokens: Output tokens reported by the provider, or ``None`` when
            the provider omitted usage.
        finish_status: Provider finish status, such as ``"completed"``.
        refusal: Refusal text when the provider returned one.
        provider_request_id: Provider response ID for this generation.
    """

    answer: str
    provider: str
    model: str
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    finish_status: str | None = None
    refusal: str | None = None
    provider_request_id: str | None = None


class LLMClient(Protocol):
    """Interface implemented by language model generation clients.

    Implementations return provider-reported token usage with each answer.
    """

    def generate(self, question: str) -> GenerationResult:
        """Generate an answer for a question.

        Args:
            question: User question to answer.

        Returns:
            Generated text with provider, model, and token usage metadata.
        """
        ...


class FakeLLMClient:
    """Return deterministic responses without contacting a model provider.

    Attributes:
        model: Model identifier reported in each generated result.
    """

    def __init__(self, model: str = "fake") -> None:
        """Initialize the fake client.

        Args:
            model: Model identifier to report in generated results.
        """
        self.model = model
        self.provider = "fake"

    def generate(self, question: str) -> GenerationResult:
        """Return a stable answer without making a provider request.

        Args:
            question: User question to answer.

        Returns:
            A deterministic answer with zero token usage.
        """
        return GenerationResult(
            answer=f"Fake answer to: {question}",
            provider="fake",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
        )


class MissingAPIKeyError(RuntimeError):
    """A provider client was used without its required API key configured."""


class OpenAIResponsesClient:
    """Generate text with OpenAI's official Python SDK and Responses API.

    The SDK client is created on first use, so the application can start
    without ``OPENAI_API_KEY`` and only fails when generation is attempted.

    SDK retries are disabled. This client retries rate limits and selected 5xx
    responses up to ``max_retries``. Timeout and connection failures are retried
    at most once, and no retry starts after the wall-clock budget is exhausted.
    Authentication failures and bad requests are not retried.
    ``responses.create`` is not idempotent, so a transport retry can bill a
    second generation when an earlier request reached the provider; those
    retries are logged with ``possible_duplicate_generation``.

    Attributes:
        _client: SDK client used to make Responses API requests, if created.
        _model: Model identifier passed to each request.
        _timeout_seconds: Per-request timeout used when creating the SDK client.
        _max_output_tokens: Optional completion-token cap sent on each request.
        _instructions: Optional instructions sent separately from the question.
        _max_retries: Extra attempts after the first request.
        _sleeper: Called with the backoff delay before each retry.
        _rng: Returns a jitter factor in the range 0 to 1.
        _clock: Monotonic clock used for the retry wall-clock budget.
    """

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
        sleeper: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the adapter with a model and optional SDK client.

        Args:
            model: OpenAI model identifier to use for requests.
            api_key: Optional key from application settings.
            client: Optional OpenAI SDK client, useful for isolated tests.
            timeout_seconds: Per-request timeout in seconds for a client this
                adapter creates. Ignored when ``client`` is injected.
            max_output_tokens: Optional completion-token cap. Omitted from the
                request when ``None``.
            instructions: Optional instructions sent separately from the
                question. Omitted from the request when ``None``.
            max_retries: Extra attempts after the first request. Zero disables
                application retries. Transport failures still cap at one retry.
            sleeper: Wait function used between retries. Tests can replace it.
            rng: Jitter source used for backoff. Tests can replace it.
            clock: Monotonic clock used for the retry wall-clock budget.

        Raises:
            ValueError: If ``max_retries`` is negative.
        """
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
        """Return the SDK client, creating it on first use.

        Returns:
            The OpenAI SDK client.

        Raises:
            MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set.
        """
        if self._client is None:
            if not self._api_key:
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set; it is required when "
                    "GENERATION_PROVIDER is 'openai'"
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                timeout=self._timeout_seconds,
                max_retries=0,
            )
        return self._client

    def _request_body(self, question: str) -> dict[str, object]:
        """Build the Responses API request from the question and client settings.

        Args:
            question: User question sent as the request input.

        Returns:
            Request fields. Instructions and the output cap are included only
            when configured.
        """
        body: dict[str, object] = {"model": self._model, "input": question}
        if self._instructions is not None:
            body["instructions"] = self._instructions
        if self._max_output_tokens is not None:
            body["max_output_tokens"] = self._max_output_tokens
        return body

    def generate(self, question: str) -> GenerationResult:
        """Generate an answer and return provider-reported token usage.

        Args:
            question: User question to send as the Responses API input.

        Returns:
            Generated text, the model identifier, and reported token usage.

        Raises:
            MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set.
            Exception: The provider error from the final attempt. Retryable
                failures are retried up to ``max_retries`` extra times.
        """
        client = self._get_client()
        body = self._request_body(question)
        started = self._clock()
        transport_retries = 0
        for attempt in range(self._max_retries + 1):
            try:
                response = client.responses.create(**body)
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
                self._sleeper(_retry_delay_seconds(attempt, self._rng()))
                continue
            return _generation_result(response, self._model)
        raise RuntimeError("generation retry loop ended without a result")


def _generation_result(response: object, fallback_model: str) -> GenerationResult:
    """Map a Responses API response without inventing omitted metadata."""
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


def _retry_class(exc: Exception) -> str | None:
    """Classify a failure for the narrowed retry policy.

    Args:
        exc: Error raised by one generation attempt.

    Returns:
        ``"status"`` for retryable HTTP statuses, ``"transport"`` for timeouts
        and connection failures, or ``None`` when the error must not be retried.
    """
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
    """Return whether this failure may be retried under the policy."""
    return _retry_class(exc) is not None


def _retry_delay_seconds(failed_attempt: int, jitter: float) -> float:
    """Return an exponential backoff delay with equal jitter.

    Args:
        failed_attempt: Zero for the delay after the first failure.
        jitter: Factor from 0 through 1 selecting a point between half the
            ceiling and the full ceiling.

    Returns:
        Seconds to wait before the next attempt. The ceiling doubles from
        half a second and stops growing at eight seconds. The delay is always
        in ``[ceiling / 2, ceiling]``.
    """
    bounded_jitter = min(1.0, max(0.0, jitter))
    ceiling = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2**failed_attempt))
    return ceiling * (0.5 + 0.5 * bounded_jitter)


def _reported_token_count(usage: object | None, name: str) -> int | None:
    """Return one usage count only when the provider reported it.

    Args:
        usage: Provider usage object, or ``None`` when usage was omitted.
        name: Usage attribute to read, such as ``"input_tokens"``.

    Returns:
        The reported count, including zero. ``None`` when usage or the count
        was omitted.
    """
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
    """Return the provider finish status string, when the response has one."""
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
    """Return the provider response ID, when the response has one."""
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

    Args:
        response: Provider response object or mapping.
        part_type: Content part type to include, such as ``"output_text"`` or
            ``"refusal"``.
        text_field: Field on matching parts that holds the text.

    Returns:
        Non-empty text values in output order.
    """
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
    """Collect refusal text from Responses API output content."""
    refusals = _output_content_texts(response, part_type="refusal", text_field="refusal")
    if not refusals:
        return None
    return "\n".join(refusals)


def _answer_text(response: object, refusal: str | None) -> str:
    """Return output text, or the refusal when the provider sent no output text.

    Prefers the SDK ``output_text`` helper when present and non-empty. Otherwise
    gathers ``output_text`` content parts from ``output[]``, then refusal text.
    """
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
    """Read a field from a mapping or an object."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
