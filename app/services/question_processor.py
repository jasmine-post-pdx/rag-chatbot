"""Question processor port, implementations, and provider selection."""

import logging
import time
from typing import Protocol

from app.config.logging import sanitize_exception_summary
from app.domain.errors import ContextBudgetExceededError
from app.domain.questions import (
    ModelMetrics,
    QuestionResult,
    RetrievalMetrics,
)
from app.llm.generation import LLMClient, OpenAIResponsesClient
from app.services.token_budget import ContextBudgetConfig, build_token_counter

logger = logging.getLogger(__name__)


class QuestionProcessor(Protocol):
    """Produces an answer result for a question."""

    async def process(
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
    """Return the current placeholder answer until RAG adapters exist.

    When a context budget is configured, fixed input is checked before the
    placeholder result is returned so oversized requests fail closed.

    :ivar _budget: Optional context-budget configuration."""

    def __init__(self, budget: ContextBudgetConfig | None = None) -> None:
        """Initialize the placeholder processor.

        :param budget: Optional context-budget configuration. Explicit
            ``is not None`` checks are used so a custom falsy budget object is
            still honored."""
        self._budget = budget

    async def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Return a placeholder answer without calling a model provider.

        :param question_id: Identifier of the question job being processed.
        :param question: User question echoed in the placeholder answer.
        :param request_id: Optional HTTP request ID used to correlate budget logs.
        :returns: A placeholder result with empty citations and zero retrieval counts.
        :raises ContextBudgetExceededError: If configured fixed input exceeds the
            context budget."""
        if self._budget is not None:
            self._budget.allocate(question, request_id=request_id)
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
    """Answer questions by awaiting an LLM client directly, without retrieval.

    Citations are empty and retrieval counts are zero until RAG is wired up.
    When a context budget is configured, fixed input is checked and the
    assembled prompt (currently the bare question) is sent to the client.

    :ivar _client: Language model client used to generate answers.
    :ivar _budget: Optional context-budget configuration."""

    def __init__(
        self,
        client: LLMClient,
        budget: ContextBudgetConfig | None = None,
    ) -> None:
        """Initialize the processor with a generation client and optional budget.

        :param client: Language model client used to generate answers.
        :param budget: Optional context-budget configuration. Explicit
            ``is not None`` checks are used so a custom falsy budget object is
            still honored."""
        self._client = client
        self._budget = budget

    async def process(
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
        :returns: The generated answer with model usage and empty retrieval metadata.
        :raises ContextBudgetExceededError: If configured fixed input exceeds the
            context budget before the provider is called."""
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

        prompt = question
        if self._budget is not None:
            try:
                allocation = self._budget.allocate(question, request_id=request_id)
            except ContextBudgetExceededError as exc:
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
                logger.error(
                    "Generation failed: %s",
                    sanitize_exception_summary(exc, sensitive_values=(question,)),
                    extra={
                        **log_fields,
                        "elapsed_ms": elapsed_ms,
                        "error_category": "context_budget_exceeded",
                        "exception_type": type(exc).__name__,
                    },
                )
                raise
            prompt = allocation.assembled_prompt

        try:
            generation = await self._client.generate(prompt)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            error_category = _generation_error_category(exc)
            summary = sanitize_exception_summary(exc, sensitive_values=(question, prompt))
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
    max_output_tokens: int | None = 1024,
    instructions: str | None = None,
    max_retries: int = 2,
    openai_client=None,
    context_window_tokens: int = 128000,
    safety_margin_tokens: int = 256,
    budget_debug: bool = False,
) -> QuestionProcessor:
    """Return the question processor for a generation provider.

    :param provider: Generation provider name, such as ``"local"`` or ``"openai"``.
    :param model: Model identifier passed to provider-backed clients.
    :param openai_api_key: Optional OpenAI key supplied by application settings.
    :param timeout_seconds: Per-request timeout for the OpenAI client.
    :param max_output_tokens: Completion-token cap for the OpenAI client. Defaults
        to ``1024``. Also used as the output reserve for context budgeting.
    :param instructions: Optional instructions sent separately from the question
        and counted as fixed input for budgeting.
    :param max_retries: Extra OpenAI attempts after rate limits, selected server
        errors, and transport failures.
    :param openai_client: Optional shared async OpenAI SDK client owned by the
        application lifespan. When omitted, the adapter may create one lazily.
    :param context_window_tokens: Model context-window size for input budgeting.
    :param safety_margin_tokens: Extra tokens reserved when estimating capacity.
    :param budget_debug: When ``True``, log extra numeric budget fields.
    :returns: The placeholder processor for ``"local"``, or an LLM-backed processor.
    :raises UnknownGenerationProviderError: If the provider is not supported."""
    output_reserve = 1024 if max_output_tokens is None else max_output_tokens
    budget = ContextBudgetConfig(
        counter=build_token_counter(provider=provider, model=model),
        context_window_tokens=context_window_tokens,
        output_reserve_tokens=output_reserve,
        safety_margin_tokens=safety_margin_tokens,
        instructions=instructions,
        debug=budget_debug,
    )
    if provider == "local":
        return PlaceholderQuestionProcessor(budget=budget)
    if provider == "openai":
        return LLMQuestionProcessor(
            OpenAIResponsesClient(
                model,
                api_key=openai_api_key,
                client=openai_client,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                instructions=instructions,
                max_retries=max_retries,
            ),
            budget=budget,
        )
    raise UnknownGenerationProviderError(
        f"Unknown GENERATION_PROVIDER {provider!r}; expected 'local' or 'openai'"
    )


def _generation_error_category(exc: Exception) -> str:
    """Map provider exceptions to stable categories without logging request data.

    :param exc: Exception raised by a generation client.
    :returns: A stable category such as ``provider_timeout`` or ``provider_error``."""
    if isinstance(exc, ContextBudgetExceededError):
        return "context_budget_exceeded"
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return "provider_timeout"
    if "ratelimit" in name or "rate_limit" in name:
        return "provider_rate_limit"
    if "authentication" in name or "permission" in name:
        return "provider_authentication"
    return "provider_error"
