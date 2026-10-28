"""Unit tests for provider-neutral generation models and interfaces."""

import pytest
from pydantic import ValidationError

from app.models import GenerationResult, LLMClient


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
        def generate(self, question: str) -> GenerationResult:
            """Return a result for the supplied question."""
            return GenerationResult(
                answer=f"Answer to: {question}",
                provider="test-provider",
                model="test-model",
                input_tokens=3,
                output_tokens=5,
            )

    client: LLMClient = FakeClient()

    assert client.generate("What is RAG?").answer == "Answer to: What is RAG?"
