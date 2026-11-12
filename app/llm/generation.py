"""Provider-neutral interfaces and models for text generation."""

from typing import Protocol

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """Generated answer and usage reported by the model provider.

    Attributes:
        answer: Text produced by the model.
        provider: Provider identifier, such as ``"openai"`` or ``"local"``.
        model: Model identifier reported for the generation.
        input_tokens: Actual number of input tokens reported by the provider.
        output_tokens: Actual number of output tokens reported by the provider.
    """

    answer: str
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


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
        usage = response.usage
        return GenerationResult(
            answer=response.output_text,
            provider="openai",
            model=response.model or self._model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
