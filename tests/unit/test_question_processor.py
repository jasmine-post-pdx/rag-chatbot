"""Unit tests for building the configured question processor."""

import pytest

from app.domain.questions import ModelMetrics, RetrievalMetrics
from app.services.question_processor import (
    LLMQuestionProcessor,
    PlaceholderQuestionProcessor,
    UnknownGenerationProviderError,
    build_question_processor,
)
from app.services.token_budget import ApproximateTokenCounter, TiktokenTokenCounter


@pytest.mark.asyncio
async def test_local_provider_uses_placeholder_processor():
    """Verify the local provider keeps the placeholder answer behavior."""
    processor = build_question_processor("local", "stub")

    assert isinstance(processor, PlaceholderQuestionProcessor)
    assert processor._budget is not None
    assert isinstance(processor._budget.counter, ApproximateTokenCounter)
    assert processor._budget.context_window_tokens == 128000
    assert processor._budget.output_reserve_tokens == 1024
    assert processor._budget.safety_margin_tokens == 256

    result = await processor.process("req-1", "What is RAG?")
    assert result.question_id == "req-1"
    assert result.answer == "This is a placeholder answer for: What is RAG?"
    assert result.citations == []
    assert result.retrieval == RetrievalMetrics(
        documents_considered=0, chunks_returned=0
    )
    assert result.model == ModelMetrics(
        provider="local-or-api",
        name="stub",
        input_tokens=0,
        output_tokens=0,
    )


def test_openai_provider_uses_llm_processor_without_requiring_api_key(monkeypatch):
    """Verify the OpenAI processor builds even when no API key is configured."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    processor = build_question_processor("openai", "gpt-test")

    assert isinstance(processor, LLMQuestionProcessor)
    assert processor._budget is not None
    assert isinstance(processor._budget.counter, TiktokenTokenCounter)
    assert processor._budget.context_window_tokens == 128000
    assert processor._budget.output_reserve_tokens == 1024
    assert processor._budget.safety_margin_tokens == 256


def test_openai_provider_receives_generation_request_settings(monkeypatch):
    """Pass timeout, output cap, budget, and instructions through to the client."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    processor = build_question_processor(
        "openai",
        "gpt-test",
        timeout_seconds=12.5,
        max_output_tokens=256,
        instructions="Be concise.",
        max_retries=4,
        context_window_tokens=8192,
        safety_margin_tokens=64,
        budget_debug=True,
    )

    client = processor._client
    assert client._timeout_seconds == 12.5
    assert client._max_output_tokens == 256
    assert client._instructions == "Be concise."
    assert client._max_retries == 4
    assert processor._budget is not None
    assert processor._budget.context_window_tokens == 8192
    assert processor._budget.output_reserve_tokens == 256
    assert processor._budget.safety_margin_tokens == 64
    assert processor._budget.instructions == "Be concise."
    assert processor._budget.debug is True


@pytest.mark.parametrize("provider", ["anthropic", "", "OpenAI"])
def test_unknown_provider_raises(provider):
    """Reject providers without a matching processor.

    :param provider: Unsupported provider name."""
    with pytest.raises(UnknownGenerationProviderError, match="GENERATION_PROVIDER"):
        build_question_processor(provider, "any-model")
