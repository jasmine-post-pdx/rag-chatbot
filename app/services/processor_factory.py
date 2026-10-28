"""Build the configured question processor from generation settings."""

from app.models.generation import OpenAIResponsesClient
from app.services.question_answerer import QuestionAnswerer
from app.services.question_processor import (
    PlaceholderQuestionProcessor,
    QuestionProcessor,
)


class UnknownGenerationProviderError(ValueError):
    """The configured generation provider has no matching processor."""


def build_question_processor(provider: str, model: str) -> QuestionProcessor:
    """Return the question processor for a generation provider.

    Args:
        provider: Generation provider name, such as ``"local"`` or ``"openai"``.
        model: Model identifier passed to provider-backed clients.

    Returns:
        The placeholder processor for ``"local"``, or an LLM-backed answerer.

    Raises:
        UnknownGenerationProviderError: If the provider is not supported.
    """
    if provider == "local":
        return PlaceholderQuestionProcessor()
    if provider == "openai":
        return QuestionAnswerer(OpenAIResponsesClient(model))
    raise UnknownGenerationProviderError(
        f"Unknown GENERATION_PROVIDER {provider!r}; expected 'local' or 'openai'"
    )
