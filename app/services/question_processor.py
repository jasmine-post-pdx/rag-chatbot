"""Question processor port, implementations, and provider selection."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from app.config.logging import sanitize_exception_summary
from app.domain.context import RankedChunk
from app.domain.errors import ContextBudgetExceededError, StructuredOutputError
from app.domain.model_outputs import AnswerV1, ClassificationV1
from app.domain.questions import (
    ModelMetrics,
    QuestionResult,
    RetrievalMetrics,
)
from app.llm.generation import LLMClient, OpenAIResponsesClient
from app.retrieval.document_store import DocumentStore
from app.services.citations import validate_and_map_citations
from app.services.token_budget import (
    ApproximateTokenCounter,
    ContextBudgetConfig,
    build_token_counter,
)
from app.tools.executor import SafeToolExecutor
from app.tools.loop import ToolLoop
from app.tools.registry import tool_schema_strings

logger = logging.getLogger(__name__)

# One extra provider call when structured output finishes incomplete/truncated.
_STRUCTURED_INCOMPLETE_RETRIES = 1


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
    """Answer questions via schema-constrained generation, optionally with tools.

    Without a document store, requests ``AnswerV1`` once via
    ``generate_structured``. With a store, runs the bounded read-only tool loop
    and maps tool activity onto ``RetrievalMetrics``. Citations stay empty until
    Step 8 retrieval grounding. Incomplete structured finishes get one bounded
    re-request on the non-tool path; refusals and validation failures fail closed.

    When a context budget is configured, fixed input is checked before any
    provider or tool call. Tool schemas are counted when tools are enabled.

    :ivar _client: Language model client used to generate answers.
    :ivar _budget: Optional context-budget configuration.
    :ivar _document_store: Optional read-only store enabling the tool loop."""

    def __init__(
        self,
        client: LLMClient,
        budget: ContextBudgetConfig | None = None,
        *,
        document_store: DocumentStore | None = None,
        tool_max_result_chars: int = 4000,
        tool_call_timeout_seconds: float = 2.0,
        tool_search_max_limit: int = 5,
        tool_max_turns: int = 3,
        tool_max_calls: int = 6,
        tool_max_calls_per_turn: int = 3,
    ) -> None:
        """Initialize the processor with a generation client and optional tools.

        :param client: Language model client used to generate answers.
        :param budget: Optional context-budget configuration. Explicit
            ``is not None`` checks are used so a custom falsy budget object is
            still honored.
        :param document_store: When not ``None``, enable the read-only tool loop.
        :param tool_max_result_chars: Observation content character cap.
        :param tool_call_timeout_seconds: Per-tool-call deadline.
        :param tool_search_max_limit: Hard max for ``search_documents.limit``.
        :param tool_max_turns: Maximum model↔tool iterations.
        :param tool_max_calls: Maximum total tool executions per question.
        :param tool_max_calls_per_turn: Maximum proposals accepted per model turn."""
        self._client = client
        self._budget = budget
        self._document_store = document_store
        self._tool_max_result_chars = tool_max_result_chars
        self._tool_call_timeout_seconds = tool_call_timeout_seconds
        self._tool_search_max_limit = tool_search_max_limit
        self._tool_max_turns = tool_max_turns
        self._tool_max_calls = tool_max_calls
        self._tool_max_calls_per_turn = tool_max_calls_per_turn

    async def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Generate a structured answer and map it to a question result.

        :param question_id: Identifier of the question job being processed.
        :param question: User question to answer.
        :param request_id: HTTP request ID used to correlate logs.
        :returns: The generated answer with model usage and retrieval metadata.
        :raises ContextBudgetExceededError: If configured fixed input exceeds the
            context budget before the provider is called.
        :raises StructuredOutputError: If structured output is refused, remains
            incomplete after recovery, fails local validation, or tool bounds trip."""
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
        schemas: list[str] = []
        if self._document_store is not None:
            schemas = tool_schema_strings(search_max_limit=self._tool_search_max_limit)
        if self._budget is not None:
            try:
                allocation = self._budget.allocate(
                    question,
                    request_id=request_id,
                    tool_schemas=schemas,
                )
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

        if self._document_store is not None:
            return await self._process_with_tools(
                question_id=question_id,
                question=question,
                prompt=prompt,
                request_id=request_id,
                log_fields=log_fields,
                started_at=started_at,
            )

        generation = None
        incomplete_retries = 0
        while True:
            try:
                generation = await self._client.generate_structured(prompt, AnswerV1)
                break
            except StructuredOutputError as exc:
                if (
                    exc.category == "incomplete"
                    and incomplete_retries < _STRUCTURED_INCOMPLETE_RETRIES
                ):
                    incomplete_retries += 1
                    logger.warning(
                        "Retrying incomplete structured output",
                        extra={
                            **log_fields,
                            "attempt": incomplete_retries,
                            "error_category": "structured_output_incomplete",
                        },
                    )
                    continue
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
                error_category = _generation_error_category(exc)
                summary = sanitize_exception_summary(
                    exc, sensitive_values=(question, prompt)
                )
                logger.error(
                    "Generation failed: %s",
                    summary,
                    extra={
                        **log_fields,
                        "elapsed_ms": elapsed_ms,
                        "error_category": error_category,
                        "exception_type": type(exc).__name__,
                        "structured_output_category": exc.category,
                    },
                )
                raise
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
                error_category = _generation_error_category(exc)
                summary = sanitize_exception_summary(
                    exc, sensitive_values=(question, prompt)
                )
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
                "insufficient_evidence": generation.parsed.insufficient_evidence,
                "claim_count": len(generation.parsed.claims),
            },
        )
        return QuestionResult(
            question_id=question_id,
            answer=generation.parsed.answer,
            citations=[],
            retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
            model=ModelMetrics(
                provider=generation.provider,
                name=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
            ),
        )

    async def _process_with_tools(
        self,
        *,
        question_id: str,
        question: str,
        prompt: str,
        request_id: str | None,
        log_fields: dict,
        started_at: float,
    ) -> QuestionResult:
        """Run the bounded tool loop and map the result.

        :param question_id: Question job identifier.
        :param question: Original user question (for sanitized error logs).
        :param prompt: Budget-assembled prompt used as the loop user input.
        :param request_id: Optional HTTP correlation ID.
        :param log_fields: Shared structured log fields.
        :param started_at: Monotonic start time for elapsed_ms.
        :returns: Question result with tool-derived retrieval metrics.
        :raises StructuredOutputError: On tool-bound or structured failures."""
        assert self._document_store is not None
        instructions = None
        if self._budget is not None:
            instructions = self._budget.instructions
        loop = ToolLoop(
            client=self._client,
            executor=SafeToolExecutor(
                self._document_store,
                max_result_chars=self._tool_max_result_chars,
                call_timeout_seconds=self._tool_call_timeout_seconds,
                search_max_limit=self._tool_search_max_limit,
            ),
            max_turns=self._tool_max_turns,
            max_calls=self._tool_max_calls,
            max_calls_per_turn=self._tool_max_calls_per_turn,
            search_max_limit=self._tool_search_max_limit,
        )
        try:
            loop_result = await loop.run(
                prompt,
                request_id=request_id,
                instructions=instructions,
            )
        except StructuredOutputError as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            error_category = _generation_error_category(exc)
            summary = sanitize_exception_summary(
                exc, sensitive_values=(question, prompt)
            )
            logger.error(
                "Generation failed: %s",
                summary,
                extra={
                    **log_fields,
                    "elapsed_ms": elapsed_ms,
                    "error_category": error_category,
                    "exception_type": type(exc).__name__,
                    "structured_output_category": exc.category,
                },
            )
            raise
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            error_category = _generation_error_category(exc)
            summary = sanitize_exception_summary(
                exc, sensitive_values=(question, prompt)
            )
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

        generation = loop_result.structured
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
                "insufficient_evidence": generation.parsed.insufficient_evidence,
                "claim_count": len(generation.parsed.claims),
                "tool_calls_executed": loop_result.tool_calls_executed,
            },
        )
        return QuestionResult(
            question_id=question_id,
            answer=generation.parsed.answer,
            citations=[],
            retrieval=RetrievalMetrics(
                documents_considered=loop_result.documents_considered,
                chunks_returned=loop_result.chunks_returned,
            ),
            model=ModelMetrics(
                provider=generation.provider,
                name=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
            ),
        )


class GroundedQuestionProcessor:
    """Retrieve, budget, and generate a citation-validated answer (Step 8).

    Flow: optional classification filters → top-k store search → token-budget
    selection → evidence-bound structured generation → citation ID validation
    against selected chunks only. Corpus text is untrusted reference evidence.

    :ivar _client: Language model client used for structured generation.
    :ivar _document_store: Read-only store used for top-k retrieval.
    :ivar _budget: Optional context-budget configuration.
    :ivar _retrieval_top_k: Maximum chunks requested from the store.
    :ivar _classifier: Optional async classifier returning ``ClassificationV1``."""

    def __init__(
        self,
        client: LLMClient,
        document_store: DocumentStore,
        budget: ContextBudgetConfig | None = None,
        *,
        retrieval_top_k: int = 5,
        classifier: Callable[[str], Awaitable[ClassificationV1]] | None = None,
    ) -> None:
        """Store dependencies for the grounded answering path.

        :param client: Language model client used to generate structured answers.
        :param document_store: Read-only store searched for evidence chunks.
        :param budget: Optional context-budget configuration for ranked selection.
        :param retrieval_top_k: Maximum hits requested from the document store.
        :param classifier: Optional async callable that returns retrieval filters."""
        self._client = client
        self._document_store = document_store
        self._budget = budget
        self._retrieval_top_k = retrieval_top_k
        self._classifier = classifier

    async def process(
        self,
        question_id: str,
        question: str,
        *,
        request_id: str | None = None,
    ) -> QuestionResult:
        """Produce a retrieval-grounded answer with validated citations.

        :param question_id: Identifier of the question job being processed.
        :param question: User question to answer.
        :param request_id: Optional HTTP request ID used to correlate logs.
        :returns: The generated answer with citations and retrieval metadata.
        :raises ContextBudgetExceededError: If fixed input alone exceeds the budget.
        :raises StructuredOutputError: On structured or citation validation failures.
        :raises Exception: Provider failures propagate without fabricating answers."""
        started_at = time.perf_counter()
        provider = getattr(self._client, "provider", type(self._client).__name__)
        model = getattr(self._client, "model", "unknown")
        log_fields: dict = {
            "question_id": question_id,
            "provider": provider,
            "model": model,
        }
        if request_id is not None:
            log_fields["request_id"] = request_id

        framework: str | None = None
        language: str | None = None
        if self._classifier is not None:
            classification = await self._classifier(question)
            framework = classification.framework
            language = classification.language

        hits = await asyncio.to_thread(
            lambda: self._document_store.search_documents(
                question,
                framework=framework,
                language=language,
                limit=self._retrieval_top_k,
            )
        )

        ranked = [
            RankedChunk(
                source_id=hit.source_id,
                chunk_id=hit.chunk_id,
                text=hit.text,
                rank=hit.rank,
                score=hit.score,
            )
            for hit in hits
        ]

        budget = self._budget
        if budget is None:
            budget = ContextBudgetConfig(
                counter=ApproximateTokenCounter(),
                context_window_tokens=128000,
                output_reserve_tokens=1024,
                safety_margin_tokens=0,
            )

        try:
            allocation = budget.allocate(
                question,
                request_id=request_id,
                ranked_chunks=ranked,
            )
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
            generation = await self._client.generate_structured(prompt, AnswerV1)
            citations = validate_and_map_citations(
                generation.parsed,
                allocation.selected_chunks,
            )
        except StructuredOutputError as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            error_category = _generation_error_category(exc)
            summary = sanitize_exception_summary(
                exc, sensitive_values=(question, prompt, *(chunk.text for chunk in ranked))
            )
            logger.error(
                "Generation failed: %s",
                summary,
                extra={
                    **log_fields,
                    "elapsed_ms": elapsed_ms,
                    "error_category": error_category,
                    "exception_type": type(exc).__name__,
                    "structured_output_category": exc.category,
                },
            )
            raise
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            error_category = _generation_error_category(exc)
            summary = sanitize_exception_summary(
                exc, sensitive_values=(question, prompt, *(chunk.text for chunk in ranked))
            )
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
                "insufficient_evidence": generation.parsed.insufficient_evidence,
                "claim_count": len(generation.parsed.claims),
                "citation_count": len(citations),
                "documents_considered": len(hits),
                "chunks_returned": allocation.selected_count,
            },
        )
        return QuestionResult(
            question_id=question_id,
            answer=generation.parsed.answer,
            citations=citations,
            retrieval=RetrievalMetrics(
                documents_considered=len(hits),
                chunks_returned=allocation.selected_count,
            ),
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
    document_store: DocumentStore | None = None,
    tool_max_result_chars: int = 4000,
    tool_call_timeout_seconds: float = 2.0,
    tool_search_max_limit: int = 5,
    tool_max_turns: int = 3,
    tool_max_calls: int = 6,
    tool_max_calls_per_turn: int = 3,
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
    :param document_store: Optional read-only store enabling document tools on the
        OpenAI path.
    :param tool_max_result_chars: Observation content character cap.
    :param tool_call_timeout_seconds: Per-tool-call deadline.
    :param tool_search_max_limit: Hard max for ``search_documents.limit``.
    :param tool_max_turns: Maximum model↔tool iterations.
    :param tool_max_calls: Maximum total tool executions per question.
    :param tool_max_calls_per_turn: Maximum proposals accepted per model turn.
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
            document_store=document_store,
            tool_max_result_chars=tool_max_result_chars,
            tool_call_timeout_seconds=tool_call_timeout_seconds,
            tool_search_max_limit=tool_search_max_limit,
            tool_max_turns=tool_max_turns,
            tool_max_calls=tool_max_calls,
            tool_max_calls_per_turn=tool_max_calls_per_turn,
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
    if isinstance(exc, StructuredOutputError):
        return f"structured_output_{exc.category}"
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return "provider_timeout"
    if "ratelimit" in name or "rate_limit" in name:
        return "provider_rate_limit"
    if "authentication" in name or "permission" in name:
        return "provider_authentication"
    return "provider_error"
