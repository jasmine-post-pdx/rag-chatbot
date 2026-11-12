"""Unit tests for provider-neutral generation models and interfaces."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from app.llm import (
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
        self.response = SimpleNamespace(
            output_text="An OpenAI answer",
            model="gpt-test-2026",
            usage=SimpleNamespace(input_tokens=7, output_tokens=9),
            status="completed",
            id="resp_ok",
            output=[],
        )

    def create(self, **kwargs):
        """Record the request and return the configured response.

        Args:
            **kwargs: Request parameters, such as ``model`` and ``input``.

        Returns:
            An object shaped like an SDK response.
        """
        self.calls.append(kwargs)
        return self.response


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
        finish_status="completed",
        refusal=None,
        provider_request_id="resp_ok",
    )


def test_openai_client_leaves_usage_null_when_provider_omits_it():
    """Do not invent token counts when the provider sends no usage."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="An OpenAI answer",
        model="gpt-test-2026",
        usage=None,
        status="completed",
        id="resp_123",
        output=[],
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    result = client.generate("What is RAG?")

    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.finish_status == "completed"
    assert result.provider_request_id == "resp_123"
    assert result.refusal is None


def test_openai_client_maps_refusal_when_output_text_is_empty():
    """Keep refusal text and use it as the answer when no output text was returned."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=4, output_tokens=None),
        status="incomplete",
        id="resp_refused",
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(type="refusal", refusal="Cannot answer that.")]
            )
        ],
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    result = client.generate("What is RAG?")

    assert result.answer == "Cannot answer that."
    assert result.refusal == "Cannot answer that."
    assert result.finish_status == "incomplete"
    assert result.provider_request_id == "resp_refused"
    assert result.input_tokens == 4
    assert result.output_tokens is None


def test_openai_client_sends_instructions_and_output_cap_when_configured():
    """Send instructions and the output cap separately from the question."""
    responses = _FakeResponses()
    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        timeout_seconds=12.5,
        max_output_tokens=256,
        instructions="Be concise.",
    )

    client.generate("What is RAG?")

    assert responses.calls == [
        {
            "model": "gpt-test",
            "input": "What is RAG?",
            "instructions": "Be concise.",
            "max_output_tokens": 256,
        }
    ]


def test_openai_client_configures_sdk_timeout_without_sdk_retries(monkeypatch):
    """Create the SDK client with the configured timeout and no SDK retries.

    Args:
        monkeypatch: Pytest fixture used to replace the OpenAI SDK constructor.
    """
    constructed_clients = []
    responses = _FakeResponses()

    def fake_openai(**kwargs):
        """Capture SDK client options and provide a fake Responses endpoint."""
        constructed_clients.append(kwargs)
        return SimpleNamespace(responses=responses)

    openai_module = ModuleType("openai")
    openai_module.OpenAI = fake_openai
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    client = OpenAIResponsesClient(
        "gpt-test", api_key="test-key", timeout_seconds=12.5
    )

    result = client.generate("What is RAG?")

    assert result.answer == "An OpenAI answer"
    assert constructed_clients == [
        {"api_key": "test-key", "timeout": 12.5, "max_retries": 0}
    ]
