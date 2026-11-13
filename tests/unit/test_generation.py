"""Unit tests for provider-neutral generation models and interfaces."""

import logging
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

    :param token_field: Token-count field to set to an invalid negative value."""
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

            :param question: User question to answer.
            :returns: An answer echoing the question, with fixed usage metadata."""
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

    :ivar calls: Keyword arguments passed to each ``create`` call, in call order."""

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

        :param **kwargs: Request parameters, such as ``model`` and ``input``.
        :returns: An object shaped like an SDK response."""
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


def test_openai_client_reads_answer_from_output_items_when_helper_missing():
    """Fall back to output[] message text when output_text is omitted."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
        status="completed",
        id="resp_structured",
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(type="output_text", text="First part."),
                    SimpleNamespace(type="output_text", text="Second part."),
                ]
            )
        ],
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    result = client.generate("What is RAG?")

    assert result.answer == "First part.\nSecond part."
    assert result.refusal is None
    assert result.provider_request_id == "resp_structured"


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

    :param monkeypatch: Pytest fixture used to replace the OpenAI SDK constructor."""
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


class _StatusFailure(Exception):
    """Provider error carrying an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        """Store the status code used by the retry policy.

        :param status_code: HTTP status reported by the provider."""
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _ScriptedResponses:
    """Return or raise scripted Responses API outcomes in order."""

    def __init__(self, outcomes: list[object]) -> None:
        """Store the outcomes and start with no recorded calls.

        :param outcomes: Response objects or exceptions, consumed one per call."""
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        """Record the call and return or raise the next outcome."""
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _completed_response():
    """Return a successful Responses API object with reported usage."""
    return SimpleNamespace(
        output_text="An OpenAI answer",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=7, output_tokens=9),
        status="completed",
        id="resp_ok",
        output=[],
    )


def _retrying_client(responses: _ScriptedResponses, max_retries: int = 2):
    """Build a client whose retries record delays instead of sleeping.

    :param responses: Scripted Responses endpoint.
    :param max_retries: Extra attempts allowed after the first request.
    :returns: The client and the list of recorded backoff delays."""
    delays: list[float] = []
    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        max_retries=max_retries,
        sleeper=delays.append,
        rng=lambda: 1.0,
    )
    return client, delays


def test_openai_client_retries_rate_limit_then_returns_success():
    """Retry HTTP 429 once, then map the successful response."""
    responses = _ScriptedResponses([_StatusFailure(429), _completed_response()])
    client, delays = _retrying_client(responses)

    result = client.generate("What is RAG?")

    assert result.answer == "An OpenAI answer"
    assert len(responses.calls) == 2
    assert delays == [0.5]


@pytest.mark.parametrize(
    "failure",
    [
        _StatusFailure(500),
        _StatusFailure(502),
        _StatusFailure(503),
        _StatusFailure(504),
        TimeoutError("timed out"),
        ConnectionError("connection reset"),
    ],
)
def test_openai_client_retries_transient_and_transport_failures(failure):
    """Retry selected server errors, timeouts, and connection failures.

    :param failure: Retryable error raised by the first provider call."""
    responses = _ScriptedResponses([failure, _completed_response()])
    client, delays = _retrying_client(responses)

    result = client.generate("What is RAG?")

    assert result.provider_request_id == "resp_ok"
    assert len(responses.calls) == 2
    assert delays == [0.5]


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 501])
def test_openai_client_does_not_retry_client_or_unselected_errors(status_code):
    """Leave authentication, bad requests, and unselected statuses unretried.

    :param status_code: Provider status that must fail on the first attempt."""
    responses = _ScriptedResponses([_StatusFailure(status_code), _completed_response()])
    client, delays = _retrying_client(responses)

    with pytest.raises(_StatusFailure, match=f"status {status_code}"):
        client.generate("What is RAG?")

    assert len(responses.calls) == 1
    assert delays == []


def test_openai_client_stops_at_the_retry_limit():
    """Make one initial request plus the configured number of extra attempts."""
    responses = _ScriptedResponses([_StatusFailure(429)] * 4)
    client, delays = _retrying_client(responses, max_retries=2)

    with pytest.raises(_StatusFailure, match="status 429"):
        client.generate("What is RAG?")

    assert len(responses.calls) == 3
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ConnectionError("reset")])
def test_openai_client_retries_transport_failures_at_most_once(failure):
    """Timeouts and connection errors get one retry even when max_retries is higher.

    :param failure: Transport error raised on every provider call."""
    responses = _ScriptedResponses([failure, failure, failure])
    client, delays = _retrying_client(responses, max_retries=2)

    with pytest.raises(type(failure)):
        client.generate("What is RAG?")

    assert len(responses.calls) == 2
    assert delays == [0.5]


def test_openai_client_stops_retrying_when_wall_clock_budget_exceeded():
    """Do not start another attempt after the retry wall-clock budget."""
    responses = _ScriptedResponses([_StatusFailure(503), _completed_response()])
    delays: list[float] = []
    readings = iter([0.0, 60.0])
    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        max_retries=2,
        sleeper=delays.append,
        rng=lambda: 1.0,
        clock=lambda: next(readings),
    )

    with pytest.raises(_StatusFailure, match="status 503"):
        client.generate("What is RAG?")

    assert len(responses.calls) == 1
    assert delays == []


def test_openai_client_logs_possible_duplicate_on_transport_retry(caplog):
    """Mark transport retries as possible duplicate generations."""
    responses = _ScriptedResponses([TimeoutError("timed out"), _completed_response()])
    client, _delays = _retrying_client(responses)
    caplog.set_level(logging.WARNING, logger="app.llm.generation")

    result = client.generate("What is RAG?")

    assert result.provider_request_id == "resp_ok"
    retry_logs = [record for record in caplog.records if record.message == "Retrying generation"]
    assert len(retry_logs) == 1
    assert retry_logs[0].retry_class == "transport"
    assert retry_logs[0].possible_duplicate_generation is True


def test_openai_client_makes_one_attempt_when_retries_are_disabled():
    """Do not retry when the retry allowance is zero."""
    responses = _ScriptedResponses([TimeoutError("timed out"), _completed_response()])
    client, delays = _retrying_client(responses, max_retries=0)

    with pytest.raises(TimeoutError, match="timed out"):
        client.generate("What is RAG?")

    assert len(responses.calls) == 1
    assert delays == []


def test_openai_client_caps_backoff_and_omits_the_question_from_retry_logs(caplog):
    """Cap backoff at eight seconds and keep the question out of retry logs.

    :param caplog: Pytest fixture used to capture structured log records."""
    responses = _ScriptedResponses([_StatusFailure(503)] * 7)
    client, delays = _retrying_client(responses, max_retries=6)
    caplog.set_level(logging.WARNING, logger="app.llm.generation")

    with pytest.raises(_StatusFailure):
        client.generate("private prompt text")

    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    assert "private prompt text" not in caplog.text
    retry_logs = [record for record in caplog.records if record.message == "Retrying generation"]
    assert [record.attempt for record in retry_logs] == [1, 2, 3, 4, 5, 6]
    assert {record.provider_status for record in retry_logs} == {503}
    assert {record.exception_type for record in retry_logs} == {"_StatusFailure"}


@pytest.mark.parametrize(
    ("failed_attempt", "jitter", "expected"),
    [
        (0, 0.0, 0.25),
        (0, 1.0, 0.5),
        (1, 0.0, 0.5),
        (1, 1.0, 1.0),
        (4, 0.0, 4.0),
        (4, 1.0, 8.0),
        (5, 0.0, 4.0),
        (5, 1.0, 8.0),
    ],
)
def test_retry_delay_uses_equal_jitter(failed_attempt, jitter, expected):
    """Equal jitter keeps delays in [ceiling / 2, ceiling], never zero.

    :param failed_attempt: Zero-based failure count used for the backoff ceiling.
    :param jitter: Injected jitter factor from 0 through 1.
    :param expected: Delay produced by equal jitter for that attempt and factor."""
    from app.llm.generation import _retry_delay_seconds

    assert _retry_delay_seconds(failed_attempt, jitter) == expected
