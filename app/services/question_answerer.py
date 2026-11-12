"""Question processor that answers questions with a language model client."""

import logging
import time

from app.config.logging import sanitize_exception_summary
from app.domain.questions import (
    ModelMetrics,
    QuestionResult,
    RetrievalMetrics,
)
from app.llm.generation import LLMClient

logger = logging.getLogger(__name__)


class QuestionAnswerer:
    """Answer questions by calling an LLM client directly, without retrieval.

    Citations are empty and retrieval counts are zero until RAG is wired up.

    Attributes:
        _client: Language model client used to generate answers.
    """

    def __init__(self, client: LLMClient) -> None:
        """Initialize the answerer with a generation client.

        Args:
            client: Language model client used to generate answers.
        """
        self._client = client

    def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Generate an answer and map it to a question result.

        Args:
            question_id: Identifier of the question job being processed.
            question: User question to answer.
            request_id: HTTP correlation ID used to correlate logs.

        Returns:
            The generated answer with model usage and empty retrieval metadata.
        """
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


def _generation_error_category(exc: Exception) -> str:
    """Map provider exceptions to stable categories without logging request data."""
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return "provider_timeout"
    if "ratelimit" in name or "rate_limit" in name:
        return "provider_rate_limit"
    if "authentication" in name or "permission" in name:
        return "provider_authentication"
    return "provider_error"
