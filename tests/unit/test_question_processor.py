"""Unit tests for building the configured question processor."""

import pytest

from app.services.question_processor import (
    LLMQuestionProcessor,
    PlaceholderQuestionProcessor,
    UnknownGenerationProviderError,
    build_question_processor,
)


def test_local_provider_uses_placeholder_processor():
    """Verify the local provider keeps the placeholder answer behavior."""
    processor = build_question_processor("local", "stub")

    assert isinstance(processor, PlaceholderQuestionProcessor)


def test_openai_provider_uses_llm_processor_without_requiring_api_key(monkeypatch):
    """Verify the OpenAI processor builds even when no API key is configured."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    processor = build_question_processor("openai", "gpt-test")

    assert isinstance(processor, LLMQuestionProcessor)


@pytest.mark.parametrize("provider", ["anthropic", "", "OpenAI"])
def test_unknown_provider_raises(provider):
    """Reject providers without a matching processor.

    Args:
        provider: Unsupported provider name.
    """
    with pytest.raises(UnknownGenerationProviderError, match="GENERATION_PROVIDER"):
        build_question_processor(provider, "any-model")
