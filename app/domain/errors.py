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
