"""Question processor port, implementations, and provider selection."""

import logging
import time
from typing import Protocol

from app.config.logging import sanitize_exception_summary
from app.domain.questions import (
    ModelMetrics,
    QuestionResult,
    RetrievalMetrics,
)
from app.llm.generation import LLMClient, OpenAIResponsesClient

logger = logging.getLogger(__name__)


class QuestionProcessor(Protocol):
    """Produces an answer result for a question."""

    def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Produce an answer result for a question.

        :param question_id: Identifier of the question job being processed.
        :param question: User question to answer.
        :param request_id: Optional HTTP request ID used to correlate logs.
        :returns: The generated answer with retrieval and model metadata."""
        ...


class PlaceholderQuestionProcessor:
    """Return the current placeholder answer until RAG adapters exist."""

    def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Return a placeholder answer without calling a model provider.

        :param question_id: Identifier of the question job being processed.
        :param question: User question echoed in the placeholder answer.
        :param request_id: Optional HTTP request ID; unused by this processor.
        :returns: A placeholder result with empty citations and zero retrieval counts."""
        return QuestionResult(
            question_id=question_id,
            answer=f"This is a placeholder answer for: {question}",
            citations=[],
            retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
            model=ModelMetrics(
                provider="local-or-api",
                name="stub",
                input_tokens=0,
                output_tokens=0,
            ),
        )


class LLMQuestionProcessor:
    """Answer questions by calling an LLM client directly, without retrieval.

    Citations are empty and retrieval counts are zero until RAG is wired up.

    :ivar _client: Language model client used to generate answers."""

    def __init__(self, client: LLMClient) -> None:
        """Initialize the processor with a generation client.

        :param client: Language model client used to generate answers."""
        self._client = client

    def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Generate an answer and map it to a question result.

        :param question_id: Identifier of the question job being processed.
        :param question: User question to answer.
        :param request_id: HTTP request ID used to correlate logs.
        :returns: The generated answer with model usage and empty retrieval metadata."""
        started_at = time.perf_counter()
        provider = getattr(self._client, "provider", type(self._client).__name__)
        model = getattr(self._client, "model", "unknown")
        log_fields = {
            "question_id": question_id,
            "provider": provider,
            "model": model,
        }
        if request_id is not None:
            log_fields["request_id"] = request_id
        try:
            generation = self._client.generate(question)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            error_category = _generation_error_category(exc)
            summary = sanitize_exception_summary(exc, sensitive_values=(question,))
            logger.error(
                "Generation failed: %s",
                summary,
                extra={
                    **log_fields,
                    "elapsed_ms": elapsed_ms,
                    "error_category": error_category,
                    "exception_type": type(exc).__name__,
                },
            )
            raise

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        logger.info(
            "Generation completed",
            extra={
                **log_fields,
                "provider": generation.provider,
                "model": generation.model,
                "elapsed_ms": elapsed_ms,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "finish_status": generation.finish_status,
                "provider_request_id": generation.provider_request_id,
            },
        )
        return QuestionResult(
            question_id=question_id,
            answer=generation.answer,
            citations=[],
            retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
            model=ModelMetrics(
                provider=generation.provider,
                name=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
            ),
        )


class UnknownGenerationProviderError(ValueError):
    """The configured generation provider has no matching processor.

    Raised during processor selection when ``GENERATION_PROVIDER`` is unsupported.
    """


def build_question_processor(
    provider: str,
    model: str,
    openai_api_key: str | None = None,
    *,
    timeout_seconds: float = 30.0,
    max_output_tokens: int | None = None,
    instructions: str | None = None,
    max_retries: int = 2,
) -> QuestionProcessor:
    """Return the question processor for a generation provider.

    :param provider: Generation provider name, such as ``"local"`` or ``"openai"``.
    :param model: Model identifier passed to provider-backed clients.
    :param openai_api_key: Optional OpenAI key supplied by application settings.
    :param timeout_seconds: Per-request timeout for the OpenAI client.
    :param max_output_tokens: Optional completion-token cap for the OpenAI client.
    :param instructions: Optional instructions sent separately from the question.
    :param max_retries: Extra OpenAI attempts after rate limits, selected server
        errors, and transport failures.
    :returns: The placeholder processor for ``"local"``, or an LLM-backed processor.
    :raises UnknownGenerationProviderError: If the provider is not supported."""
    if provider == "local":
        return PlaceholderQuestionProcessor()
    if provider == "openai":
        return LLMQuestionProcessor(
            OpenAIResponsesClient(
                model,
                api_key=openai_api_key,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                instructions=instructions,
                max_retries=max_retries,
            )
        )
    raise UnknownGenerationProviderError(
        f"Unknown GENERATION_PROVIDER {provider!r}; expected 'local' or 'openai'"
    )


def _generation_error_category(exc: Exception) -> str:
    """Map provider exceptions to stable categories without logging request data.

    :param exc: Exception raised by a generation client.
    :returns: A stable category such as ``provider_timeout`` or ``provider_error``."""
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return "provider_timeout"
    if "ratelimit" in name or "rate_limit" in name:
        return "provider_rate_limit"
    if "authentication" in name or "permission" in name:
        return "provider_authentication"
    return "provider_error"
