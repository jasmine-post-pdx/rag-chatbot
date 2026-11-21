"""Unit tests for token-counter scaffolding."""

from app.services.token_budget import (
    ApproximateTokenCounter,
    TiktokenTokenCounter,
    build_token_counter,
)


def test_approximate_counter_is_always_an_estimate():
    """Local and test counters always label counts as estimates."""
    counter = ApproximateTokenCounter()

    assert counter.is_estimate is True
    assert counter.count("") == 0
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2


def test_build_token_counter_uses_approximate_for_local():
    """The local provider uses the character-based estimate counter.

    :returns: None."""
    counter = build_token_counter(provider="local", model="stub")

    assert isinstance(counter, ApproximateTokenCounter)
    assert counter.is_estimate is True


def test_build_token_counter_uses_tiktoken_for_openai():
    """The OpenAI provider uses a tiktoken-backed counter.

    :returns: None."""
    counter = build_token_counter(provider="openai", model="gpt-4o-mini")

    assert isinstance(counter, TiktokenTokenCounter)
    assert counter.is_estimate is False
    assert counter.count("hello") > 0


def test_tiktoken_counter_marks_unknown_models_as_estimates():
    """Unknown model names fall back to ``cl100k_base`` and mark estimates.

    :returns: None."""
    counter = TiktokenTokenCounter("not-a-real-model-xyz")

    assert counter.is_estimate is True
    assert counter.count("budget") > 0
