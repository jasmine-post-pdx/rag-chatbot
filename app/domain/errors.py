"""Domain errors shared by question processing and persistence."""


class IdempotencyKeyReuseError(ValueError):
    """An idempotency key was already used for a different question."""

    def __init__(self) -> None:
        """Initialize with a stable client-facing error message."""
        super().__init__("Idempotency-Key was already used for a different question")


class IncompleteIdempotencyRecordError(ValueError):
    """An idempotency key points to a stored record missing required fields."""

    def __init__(self) -> None:
        """Initialize with a stable client-facing error message."""
        super().__init__("Idempotency-Key record is incomplete")


class InvalidIdempotencyKeyError(ValueError):
    """An idempotency key is blank or longer than the allowed length."""

    def __init__(self) -> None:
        """Initialize with a stable client-facing error message."""
        super().__init__("Idempotency-Key must contain 1 to 255 characters")


class ContextBudgetExceededError(ValueError):
    """Fixed prompt input alone exceeds the model context budget.

    Raised before any provider call so oversized requests are never sent
    silently. The exception message is safe for logs and does not include
    question, history, tool, or chunk text.

    :ivar estimated_fixed_tokens: Tokens counted for fixed input plus reserves.
    :ivar context_window_tokens: Configured model context-window size.
    :ivar is_estimate: Whether the token counter labeled counts as estimates."""

    def __init__(
        self,
        *,
        estimated_fixed_tokens: int,
        context_window_tokens: int,
        is_estimate: bool,
    ) -> None:
        """Record budget figures without embedding prompt content.

        :param estimated_fixed_tokens: Tokens counted for fixed input plus reserves.
        :param context_window_tokens: Configured model context-window size.
        :param is_estimate: Whether the underlying token counts are estimates."""
        self.estimated_fixed_tokens = estimated_fixed_tokens
        self.context_window_tokens = context_window_tokens
        self.is_estimate = is_estimate
        super().__init__(
            "Fixed input exceeds the model context budget "
            f"({estimated_fixed_tokens} > {context_window_tokens}"
            f"{', estimate' if is_estimate else ''})"
        )


class StructuredOutputError(ValueError):
    """Structured model output was refused, incomplete, invalid, or unsupported.

    The exception message is safe for logs and does not include prompt, answer,
    claim, or document text.

    :ivar category: Stable failure category such as ``refusal``, ``incomplete``,
        ``validation``, or ``unsupported``."""

    def __init__(self, category: str, message: str | None = None) -> None:
        """Record a safe category and optional short message.

        :param category: Stable failure category for logging and mapping.
        :param message: Optional safe detail; defaults to a category-based text."""
        self.category = category
        if message is None:
            message = f"Structured model output failed: {category}"
        super().__init__(message)
