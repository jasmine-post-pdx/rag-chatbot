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


class OpenAIResponsesClient:
    """Generate text with OpenAI's official Python SDK and Responses API.

    Attributes:
        _client: SDK client used to make Responses API requests.
        _model: Model identifier passed to each request.
    """

    def __init__(self, model: str, client=None) -> None:
        """Initialize the adapter with a model and optional SDK client.

        Args:
            model: OpenAI model identifier to use for requests.
            client: Optional OpenAI SDK client, useful for isolated tests.
        """
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._model = model

    def generate(self, question: str) -> GenerationResult:
        """Generate an answer and return provider-reported token usage.

        Args:
            question: User question to send as the Responses API input.

        Returns:
            Generated text, the model identifier, and reported token usage.
        """
        response = self._client.responses.create(model=self._model, input=question)
        usage = response.usage
        return GenerationResult(
            answer=response.output_text,
            provider="openai",
            model=response.model or self._model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
