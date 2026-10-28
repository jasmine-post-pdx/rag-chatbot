"""Build the configured question processor from generation settings."""

from app.models.generation import OpenAIResponsesClient
from app.services.question_answerer import QuestionAnswerer
from app.services.question_processor import (
    PlaceholderQuestionProcessor,
    QuestionProcessor,
)


class UnknownGenerationProviderError(ValueError):
    """The configured generation provider has no matching processor."""


def build_question_processor(
    provider: str, model: str, openai_api_key: str | None = None
) -> QuestionProcessor:
    """Return the question processor for a generation provider.

    Args:
        provider: Generation provider name, such as ``"local"`` or ``"openai"``.
        model: Model identifier passed to provider-backed clients.
        openai_api_key: Optional OpenAI key supplied by application settings.

    Returns:
        The placeholder processor for ``"local"``, or an LLM-backed answerer.

    Raises:
        UnknownGenerationProviderError: If the provider is not supported.
    """
    if provider == "local":
        return PlaceholderQuestionProcessor()
    if provider == "openai":
        return QuestionAnswerer(OpenAIResponsesClient(model, api_key=openai_api_key))
    raise UnknownGenerationProviderError(
        f"Unknown GENERATION_PROVIDER {provider!r}; expected 'local' or 'openai'"
    )
