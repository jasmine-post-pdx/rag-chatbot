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
