"""Provider-neutral interfaces and models for text generation."""

from typing import Protocol

from pydantic import BaseModel, Field


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

    SDK retries are disabled. Request retries belong to the application so
    they are not stacked on top of the SDK's own retries.

    Attributes:
        _client: SDK client used to make Responses API requests, if created.
        _model: Model identifier passed to each request.
        _timeout_seconds: Per-request timeout used when creating the SDK client.
        _max_output_tokens: Optional completion-token cap sent on each request.
        _instructions: Optional instructions sent separately from the question.
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
        """
        self._client = client
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._instructions = instructions
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
        """
        response = self._get_client().responses.create(**self._request_body(question))
        refusal = _refusal_text(response)
        reported_model = getattr(response, "model", None)
        if not isinstance(reported_model, str) or reported_model == "":
            reported_model = self._model
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


def _refusal_text(response: object) -> str | None:
    """Collect refusal text from Responses API output content."""
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if output is None:
        return None
    refusals: list[str] = []
    for item in output:
        content = _mapping_or_attr(item, "content") or []
        for part in content:
            if _mapping_or_attr(part, "type") != "refusal":
                continue
            text = _mapping_or_attr(part, "refusal")
            if isinstance(text, str) and text:
                refusals.append(text)
    if not refusals:
        return None
    return "\n".join(refusals)


def _answer_text(response: object, refusal: str | None) -> str:
    """Return output text, or the refusal when the provider sent no output text."""
    output_text = getattr(response, "output_text", None)
    if output_text is None and isinstance(response, dict):
        output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text != "":
        return output_text
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
