"""Token counting and deterministic context-budget allocation.

Exact provider accounting is not always available. Counters expose
``is_estimate`` so callers can label budget results honestly and apply a
configurable safety margin.

Oversized-chunk policy: ranked chunks are considered in order. Any chunk that
does not fit the remaining retrieval capacity (including framing overhead when
the first chunk is selected) is skipped. Citation identifiers are preserved
only for selected chunks.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from app.domain.context import BudgetAllocation, BudgetRequest, RankedChunk
from app.domain.errors import ContextBudgetExceededError

logger = logging.getLogger(__name__)

# Models whose tiktoken encoding is treated as exact enough to not mark estimates.
_KNOWN_TIKTOKEN_MODELS: frozenset[str] = frozenset(
    {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }
)

# Request framing wrapped around selected evidence. Counted once when any chunk
# is selected; empty-chunk prompts keep the bare question for the current API.
_FRAMING_PREFIX = "Evidence:\n"
_FRAMING_SUFFIX = "\n\nQuestion:\n"
_CHUNK_SEPARATOR = "\n\n"


class TokenCounter(Protocol):
    """Count tokens in text for context budgeting.

    :ivar is_estimate: ``True`` when counts are approximate rather than an
        exact encoding for the configured model."""

    is_estimate: bool

    def count(self, text: str) -> int:
        """Return the token count for ``text``.

        :param text: Text to measure.
        :returns: Non-negative token count."""
        ...


class ApproximateTokenCounter:
    """Estimate tokens as ``ceil(len(text) / 4)``.

    Used for the local placeholder path and deterministic unit tests when a
    model-specific tokenizer is unavailable.

    :ivar is_estimate: Always ``True`` for this counter."""

    is_estimate = True

    def count(self, text: str) -> int:
        """Return a character-based token estimate.

        :param text: Text to measure.
        :returns: ``ceil(len(text) / 4)``, or ``0`` for empty text."""
        if not text:
            return 0
        length = len(text)
        return (length + 3) // 4


class TiktokenTokenCounter:
    """Count tokens with tiktoken for OpenAI-oriented models.

    Known model names use ``tiktoken.encoding_for_model``. Unknown models fall
    back to the ``cl100k_base`` encoding and set ``is_estimate`` to ``True``.

    :ivar is_estimate: ``False`` for known model encodings; ``True`` when the
        fallback encoding is used.
    :ivar _encoding: Loaded tiktoken encoding used for counting."""

    def __init__(self, model: str) -> None:
        """Load an encoding for ``model``, falling back to ``cl100k_base``.

        :param model: Generation model identifier from application settings."""
        import tiktoken

        normalized = model.strip()
        if normalized in _KNOWN_TIKTOKEN_MODELS:
            try:
                self._encoding = tiktoken.encoding_for_model(normalized)
                self.is_estimate = False
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self.is_estimate = True
        else:
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self.is_estimate = True

    def count(self, text: str) -> int:
        """Return the tiktoken token count for ``text``.

        :param text: Text to measure.
        :returns: Number of tokens in the encoding, or ``0`` for empty text."""
        if not text:
            return 0
        return len(self._encoding.encode(text))


def build_token_counter(*, provider: str, model: str) -> TokenCounter:
    """Return a token counter appropriate for the generation provider.

    :param provider: Generation provider name, such as ``"local"`` or ``"openai"``.
    :param model: Model identifier used to select a tiktoken encoding.
    :returns: An approximate counter for ``local``, otherwise a tiktoken counter."""
    if provider == "local":
        return ApproximateTokenCounter()
    return TiktokenTokenCounter(model)


class ContextBudgetConfig:
    """Settings and counter used to allocate context before generation.

    :ivar counter: Token counter used for all budget measurements.
    :ivar context_window_tokens: Model context-window size.
    :ivar output_reserve_tokens: Tokens reserved for model output.
    :ivar safety_margin_tokens: Extra tokens reserved for estimate error.
    :ivar instructions: Optional instructions counted as fixed input.
    :ivar debug: When ``True``, log extra numeric budget fields."""

    def __init__(
        self,
        *,
        counter: TokenCounter,
        context_window_tokens: int,
        output_reserve_tokens: int,
        safety_margin_tokens: int,
        instructions: str | None = None,
        debug: bool = False,
    ) -> None:
        """Store budget settings for later allocation calls.

        :param counter: Token counter used for all budget measurements.
        :param context_window_tokens: Model context-window size.
        :param output_reserve_tokens: Tokens reserved for model output.
        :param safety_margin_tokens: Extra tokens reserved for estimate error.
        :param instructions: Optional instructions counted as fixed input.
        :param debug: When ``True``, log extra numeric budget fields."""
        self.counter = counter
        self.context_window_tokens = context_window_tokens
        self.output_reserve_tokens = output_reserve_tokens
        self.safety_margin_tokens = safety_margin_tokens
        self.instructions = instructions
        self.debug = debug

    def allocate(
        self,
        question: str,
        *,
        request_id: str | None = None,
        ranked_chunks: Sequence[RankedChunk] = (),
        history: Sequence[str] = (),
        tool_schemas: Sequence[str] = (),
    ) -> BudgetAllocation:
        """Allocate context for a question and log safe budget metadata.

        :param question: User question text.
        :param request_id: Optional HTTP correlation ID for logs.
        :param ranked_chunks: Retrieval candidates in preferred order.
        :param history: Prior turn texts counted as fixed input.
        :param tool_schemas: Serialized tool schema strings.
        :returns: The deterministic budget allocation.
        :raises ContextBudgetExceededError: If fixed input alone exceeds the window."""
        allocation = allocate_context_budget(
            BudgetRequest(
                question=question,
                instructions=self.instructions,
                history=list(history),
                tool_schemas=list(tool_schemas),
                ranked_chunks=list(ranked_chunks),
            ),
            counter=self.counter,
            context_window_tokens=self.context_window_tokens,
            output_reserve_tokens=self.output_reserve_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
        )
        log_budget_allocation(
            allocation,
            request_id=request_id,
            debug=self.debug,
        )
        return allocation


def allocate_context_budget(
    request: BudgetRequest,
    *,
    counter: TokenCounter,
    context_window_tokens: int,
    output_reserve_tokens: int,
    safety_margin_tokens: int,
) -> BudgetAllocation:
    """Allocate ranked chunks into the remaining context capacity.

    Fixed input is the sum of instructions, question, history, tool schemas,
    the output reserve, and the safety margin. Available retrieval capacity is
    the context window minus that fixed cost. Framing tokens for the evidence
    wrapper are taken from retrieval capacity when at least one chunk is
    selected.

    Chunks are considered in the order supplied. A chunk that does not fit the
    remaining capacity is skipped so later, smaller chunks may still be
    selected. If fixed input alone exceeds the window, raises
    ``ContextBudgetExceededError`` and does not assemble a provider prompt.

    :param request: Question, optional instructions/history/tools, and chunks.
    :param counter: Token counter used for all measurements.
    :param context_window_tokens: Model context-window size.
    :param output_reserve_tokens: Tokens reserved for model output.
    :param safety_margin_tokens: Extra tokens reserved for estimate error.
    :returns: Selected chunks, counts, and the assembled generation prompt.
    :raises ContextBudgetExceededError: If fixed input alone exceeds the window.
    :raises ValueError: If window, output reserve, or safety margin are invalid."""
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be greater than 0")
    if output_reserve_tokens <= 0:
        raise ValueError("output_reserve_tokens must be greater than 0")
    if safety_margin_tokens < 0:
        raise ValueError("safety_margin_tokens must be zero or greater")

    instructions_tokens = (
        counter.count(request.instructions) if request.instructions is not None else 0
    )
    question_tokens = counter.count(request.question)
    history_tokens = sum(counter.count(item) for item in request.history)
    tool_tokens = sum(counter.count(schema) for schema in request.tool_schemas)
    estimated_fixed_tokens = (
        instructions_tokens
        + question_tokens
        + history_tokens
        + tool_tokens
        + output_reserve_tokens
        + safety_margin_tokens
    )
    is_estimate = counter.is_estimate

    if estimated_fixed_tokens > context_window_tokens:
        raise ContextBudgetExceededError(
            estimated_fixed_tokens=estimated_fixed_tokens,
            context_window_tokens=context_window_tokens,
            is_estimate=is_estimate,
        )

    available_retrieval_tokens = context_window_tokens - estimated_fixed_tokens
    framing_tokens = counter.count(_FRAMING_PREFIX) + counter.count(_FRAMING_SUFFIX)
    separator_tokens = counter.count(_CHUNK_SEPARATOR)

    selected: list[RankedChunk] = []
    remaining = available_retrieval_tokens
    framing_applied = False
    discarded_count = 0

    for chunk in request.ranked_chunks:
        chunk_tokens = counter.count(chunk.text)
        needed = chunk_tokens
        if not framing_applied:
            needed += framing_tokens
        elif selected:
            needed += separator_tokens
        if needed > remaining:
            discarded_count += 1
            continue
        if not framing_applied:
            remaining -= framing_tokens
            framing_applied = True
        elif selected:
            remaining -= separator_tokens
        remaining -= chunk_tokens
        selected.append(chunk)

    assembled_prompt = _assemble_prompt(request.question, selected)
    prompt_tokens = counter.count(assembled_prompt) + instructions_tokens
    # History and tool schemas are counted in the fixed budget and are expected
    # to be supplied by the provider path separately when those features exist.
    # Include them in estimated_prompt_tokens so callers can verify the full
    # assembled input stays within the window after reserves.
    estimated_prompt_tokens = prompt_tokens + history_tokens + tool_tokens

    return BudgetAllocation(
        selected_chunks=selected,
        eligible_count=len(request.ranked_chunks),
        selected_count=len(selected),
        discarded_count=discarded_count,
        available_retrieval_tokens=available_retrieval_tokens,
        estimated_fixed_tokens=estimated_fixed_tokens,
        estimated_prompt_tokens=estimated_prompt_tokens,
        is_estimate=is_estimate,
        assembled_prompt=assembled_prompt,
    )


def _assemble_prompt(question: str, selected_chunks: Sequence[RankedChunk]) -> str:
    """Build the generation input from the question and selected evidence.

    :param question: User question text.
    :param selected_chunks: Chunks included in the budget allocation.
    :returns: The bare question when no chunks were selected; otherwise the
        framed evidence block followed by the question."""
    if not selected_chunks:
        return question
    evidence = _CHUNK_SEPARATOR.join(chunk.text for chunk in selected_chunks)
    return f"{_FRAMING_PREFIX}{evidence}{_FRAMING_SUFFIX}{question}"


def log_budget_allocation(
    allocation: BudgetAllocation,
    *,
    request_id: str | None = None,
    debug: bool = False,
) -> None:
    """Log safe budget metadata without prompt, history, tool, or chunk text.

    :param allocation: Result of ``allocate_context_budget``.
    :param request_id: Optional HTTP correlation ID.
    :param debug: When ``True``, include additional numeric budget fields."""
    extra: dict[str, object] = {
        "eligible_count": allocation.eligible_count,
        "selected_count": allocation.selected_count,
        "discarded_count": allocation.discarded_count,
        "is_estimate": allocation.is_estimate,
        "error_category": "context_budget",
    }
    if request_id is not None:
        extra["request_id"] = request_id
    if debug:
        extra.update(
            {
                "available_retrieval_tokens": allocation.available_retrieval_tokens,
                "estimated_fixed_tokens": allocation.estimated_fixed_tokens,
                "estimated_prompt_tokens": allocation.estimated_prompt_tokens,
            }
        )
    logger.info("Context budget allocated", extra=extra)


def assert_allocation_within_budget(
    allocation: BudgetAllocation,
    *,
    context_window_tokens: int,
    output_reserve_tokens: int,
    safety_margin_tokens: int,
) -> None:
    """Verify assembled prompt tokens fit under the window after reserves.

    :param allocation: Result of ``allocate_context_budget``.
    :param context_window_tokens: Model context-window size.
    :param output_reserve_tokens: Tokens reserved for model output.
    :param safety_margin_tokens: Extra tokens reserved for estimate error.
    :raises AssertionError: If the assembled input exceeds the allowed budget."""
    allowed = context_window_tokens - output_reserve_tokens - safety_margin_tokens
    assert allocation.estimated_prompt_tokens <= allowed, (
        f"assembled input {allocation.estimated_prompt_tokens} exceeds "
        f"allowed budget {allowed}"
    )
