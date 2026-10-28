"""Unit tests for provider-neutral generation models and interfaces."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models import (
    GenerationResult,
    LLMClient,
    MissingAPIKeyError,
    OpenAIResponsesClient,
)


def test_generation_result_carries_answer_provider_model_and_usage():
    """Verify generation results carry the answer and provider usage metadata."""
    result = GenerationResult(
        answer="A generated answer",
        provider="test-provider",
        model="test-model",
        input_tokens=12,
        output_tokens=34,
    )

    assert result.answer == "A generated answer"
    assert result.provider == "test-provider"
    assert result.model == "test-model"
    assert result.input_tokens == 12
    assert result.output_tokens == 34


@pytest.mark.parametrize("token_field", ["input_tokens", "output_tokens"])
def test_generation_result_rejects_negative_token_usage(token_field):
    """Reject negative token counts for either usage field.

    Args:
        token_field: Token-count field to set to an invalid negative value.
    """
    with pytest.raises(ValidationError):
        GenerationResult(
            answer="A generated answer",
            provider="test-provider",
            model="test-model",
            input_tokens=1 if token_field != "input_tokens" else -1,
            output_tokens=1 if token_field != "output_tokens" else -1,
        )


def test_llm_client_uses_structural_generate_interface():
    """Verify a client can satisfy the Protocol without explicit inheritance."""
    class FakeClient:
        """Satisfy ``LLMClient`` structurally, without inheriting from it."""

        def generate(self, question: str) -> GenerationResult:
            """Return a result for the supplied question.

            Args:
                question: User question to answer.

            Returns:
                An answer echoing the question, with fixed usage metadata.
            """
            return GenerationResult(
                answer=f"Answer to: {question}",
                provider="test-provider",
                model="test-model",
                input_tokens=3,
                output_tokens=5,
            )

    client: LLMClient = FakeClient()

    assert client.generate("What is RAG?").answer == "Answer to: What is RAG?"


class _FakeResponses:
    """Record Responses API calls and return a canned response.

    Attributes:
        calls: Keyword arguments passed to each ``create`` call, in call order.
    """

    def __init__(self) -> None:
        """Initialize the fake with no recorded calls."""
        self.calls: list[dict] = []

    def create(self, **kwargs):
        """Record the request and return a response with usage.

        Args:
            **kwargs: Request parameters, such as ``model`` and ``input``.

        Returns:
            An object shaped like an SDK response, with output text, model, and usage.
        """
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="An OpenAI answer",
            model="gpt-test-2026",
            usage=SimpleNamespace(input_tokens=7, output_tokens=9),
        )


def test_openai_client_can_be_constructed_without_api_key(monkeypatch):
    """Verify a missing API key does not fail until generation is attempted."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    OpenAIResponsesClient("gpt-test")


def test_openai_client_raises_clear_error_when_used_without_api_key(monkeypatch):
    """Verify generation without an API key fails with an actionable message."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIResponsesClient("gpt-test")

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is not set"):
        client.generate("What is RAG?")


def test_openai_client_maps_response_using_injected_sdk_client(monkeypatch):
    """Verify an injected SDK client is used and its response is mapped."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    responses = _FakeResponses()
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    result = client.generate("What is RAG?")

    assert responses.calls == [{"model": "gpt-test", "input": "What is RAG?"}]
    assert result == GenerationResult(
        answer="An OpenAI answer",
        provider="openai",
        model="gpt-test-2026",
        input_tokens=7,
        output_tokens=9,
    )
