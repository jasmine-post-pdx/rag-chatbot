"""Token counting and context-budget scaffolding for generation requests.

Exact provider accounting is not always available. Counters expose
``is_estimate`` so callers can label budget results honestly and apply a
configurable safety margin.
"""

from typing import Protocol

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
