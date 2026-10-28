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

    Attributes:
        _client: SDK client used to make Responses API requests, if created.
        _model: Model identifier passed to each request.
    """

    def __init__(self, model: str, api_key: str | None = None, client=None) -> None:
        """Initialize the adapter with a model and optional SDK client.

        Args:
            model: OpenAI model identifier to use for requests.
            api_key: Optional key from application settings.
            client: Optional OpenAI SDK client, useful for isolated tests.
        """
        self._client = client
        self._model = model
        self._api_key = api_key
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
                timeout=30.0,
                max_retries=2,
            )
        return self._client

    def generate(self, question: str) -> GenerationResult:
        """Generate an answer and return provider-reported token usage.

        Args:
            question: User question to send as the Responses API input.

        Returns:
            Generated text, the model identifier, and reported token usage.

        Raises:
            MissingAPIKeyError: If ``OPENAI_API_KEY`` is not set.
        """
        response = self._get_client().responses.create(model=self._model, input=question)
        usage = response.usage
        return GenerationResult(
            answer=response.output_text,
            provider="openai",
            model=response.model or self._model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
