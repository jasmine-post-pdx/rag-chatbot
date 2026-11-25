"""Unit tests for provider-neutral generation models and interfaces."""

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from app.domain.errors import StructuredOutputError
from app.domain.model_outputs import AnswerClaim, AnswerV1
from app.domain.tools import ToolCallProposal
from app.llm import (
    FakeLLMClient,
    GenerationResult,
    LLMClient,
    MissingAPIKeyError,
    OpenAIResponsesClient,
    StructuredGenerationResult,
    ToolTurnResult,
)
from app.tools.registry import tool_definitions


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


@pytest.mark.asyncio
async def test_llm_client_uses_structural_generate_interface():
    """Verify a client can satisfy the Protocol without explicit inheritance."""

    class FakeClient:
        """Satisfy ``LLMClient`` structurally, without inheriting from it."""

        async def generate(self, question: str) -> GenerationResult:
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

        async def generate_structured(self, prompt: str, output_model: type):
            """Return a canned structured answer for protocol satisfaction."""
            return StructuredGenerationResult(
                parsed=output_model(
                    answer=f"Structured: {prompt}",
                    claims=[],
                    insufficient_evidence=True,
                ),
                provider="test-provider",
                model="test-model",
                input_tokens=3,
                output_tokens=5,
            )

        async def generate_tool_turn(
            self,
            input_items,
            tools,
            output_model: type,
        ):
            """Return a final structured tool turn for protocol satisfaction."""
            return ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=output_model(
                        answer="Final",
                        claims=[],
                        insufficient_evidence=True,
                    ),
                    provider="test-provider",
                    model="test-model",
                    input_tokens=1,
                    output_tokens=1,
                ),
            )

    client: LLMClient = FakeClient()

    result = await client.generate("What is RAG?")
    assert result == GenerationResult(
        answer="Answer to: What is RAG?",
        provider="test-provider",
        model="test-model",
        input_tokens=3,
        output_tokens=5,
    )

    structured = await client.generate_structured("What is RAG?", AnswerV1)
    assert structured.parsed == AnswerV1(
        answer="Structured: What is RAG?",
        claims=[],
        insufficient_evidence=True,
    )
    assert structured.provider == "test-provider"
    assert structured.model == "test-model"
    assert structured.input_tokens == 3
    assert structured.output_tokens == 5

    turn = await client.generate_tool_turn("q", tools=[], output_model=AnswerV1)
    assert turn.kind == "structured"
    assert turn.structured is not None
    assert turn.structured.parsed.answer == "Final"


def _valid_answer_payload() -> dict:
    """Return a minimal valid AnswerV1 payload."""
    return {
        "schema_version": "1",
        "answer": "Structured fake answer",
        "claims": [
            {"claim": "A supported claim", "citation_chunk_ids": ["chunk-1"]},
        ],
        "insufficient_evidence": False,
    }


@pytest.mark.asyncio
async def test_fake_client_generate_structured_success():
    """Return and locally validate a canned structured payload."""
    client = FakeLLMClient(structured_payload=_valid_answer_payload())

    result = await client.generate_structured("What is RAG?", AnswerV1)

    assert isinstance(result.parsed, AnswerV1)
    assert result.parsed.answer == "Structured fake answer"
    assert result.provider == "fake"
    assert result.input_tokens == 0
    assert result.finish_status == "completed"


@pytest.mark.asyncio
async def test_fake_client_generate_structured_refusal():
    """Surface refusals as StructuredOutputError without inventing fields."""
    client = FakeLLMClient(structured_refusal="Cannot answer that.")

    with pytest.raises(StructuredOutputError, match="refusal") as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "refusal"


@pytest.mark.asyncio
async def test_fake_client_generate_structured_incomplete():
    """Surface incomplete structured finishes as StructuredOutputError."""
    client = FakeLLMClient(structured_incomplete=True)

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "incomplete"


@pytest.mark.asyncio
async def test_fake_client_generate_structured_validation_failure():
    """Reject canned payloads that violate AnswerV1 invariants."""
    client = FakeLLMClient(
        structured_payload={
            "schema_version": "1",
            "answer": "Bad",
            "claims": [{"claim": "Unsupported", "citation_chunk_ids": []}],
            "insufficient_evidence": False,
        }
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "validation"


class _FakeResponses:
    """Record Responses API calls and return a canned response.

    :ivar calls: Keyword arguments passed to each ``create`` or ``parse`` call."""

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
            output_parsed=None,
        )

    async def create(self, **kwargs):
        """Record the request and return the configured response.

        :param **kwargs: Request parameters, such as ``model`` and ``input``.
        :returns: An object shaped like an SDK response."""
        self.calls.append(kwargs)
        return self.response

    async def parse(self, **kwargs):
        """Record a structured parse request and return the configured response.

        :param **kwargs: Request parameters, including ``text_format``.
        :returns: An object shaped like an SDK parsed response."""
        self.calls.append(kwargs)
        return self.response


def test_openai_client_can_be_constructed_without_api_key(monkeypatch):
    """Verify a missing API key does not fail until generation is attempted."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    OpenAIResponsesClient("gpt-test")


@pytest.mark.asyncio
async def test_openai_client_raises_clear_error_when_used_without_api_key(monkeypatch):
    """Verify generation without an API key fails with an actionable message."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIResponsesClient("gpt-test")

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is not set"):
        await client.generate("What is RAG?")


@pytest.mark.asyncio
async def test_openai_client_maps_response_using_injected_sdk_client(monkeypatch):
    """Verify an injected SDK client is used and its response is mapped."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    responses = _FakeResponses()
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    result = await client.generate("What is RAG?")

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


@pytest.mark.asyncio
async def test_openai_client_leaves_usage_null_when_provider_omits_it():
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

    result = await client.generate("What is RAG?")

    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.finish_status == "completed"
    assert result.provider_request_id == "resp_123"
    assert result.refusal is None


@pytest.mark.asyncio
async def test_openai_client_maps_refusal_when_output_text_is_empty():
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

    result = await client.generate("What is RAG?")

    assert result.answer == "Cannot answer that."
    assert result.refusal == "Cannot answer that."
    assert result.finish_status == "incomplete"
    assert result.provider_request_id == "resp_refused"
    assert result.input_tokens == 4
    assert result.output_tokens is None


@pytest.mark.asyncio
async def test_openai_client_reads_answer_from_output_items_when_helper_missing():
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

    result = await client.generate("What is RAG?")

    assert result.answer == "First part.\nSecond part."
    assert result.refusal is None
    assert result.provider_request_id == "resp_structured"


@pytest.mark.asyncio
async def test_openai_client_sends_instructions_and_output_cap_when_configured():
    """Send instructions and the output cap separately from the question."""
    responses = _FakeResponses()
    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        timeout_seconds=12.5,
        max_output_tokens=256,
        instructions="Be concise.",
    )

    await client.generate("What is RAG?")

    assert responses.calls == [
        {
            "model": "gpt-test",
            "input": "What is RAG?",
            "instructions": "Be concise.",
            "max_output_tokens": 256,
        }
    ]


@pytest.mark.asyncio
async def test_openai_client_generate_structured_uses_parse_and_validates():
    """Pass text_format to responses.parse and re-validate the parsed payload."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=11, output_tokens=13),
        status="completed",
        id="resp_parsed",
        output=[],
        output_parsed=AnswerV1(
            answer="Parsed answer",
            claims=[AnswerClaim(claim="Claim", citation_chunk_ids=["c1"])],
            insufficient_evidence=False,
        ),
    )
    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        instructions="Be precise.",
        max_output_tokens=128,
    )

    result = await client.generate_structured("What is RAG?", AnswerV1)

    assert responses.calls == [
        {
            "model": "gpt-test",
            "input": "What is RAG?",
            "instructions": "Be precise.",
            "max_output_tokens": 128,
            "text_format": AnswerV1,
        }
    ]
    assert result.parsed.answer == "Parsed answer"
    assert result.provider_request_id == "resp_parsed"
    assert result.input_tokens == 11
    assert result.output_tokens == 13


@pytest.mark.asyncio
async def test_openai_client_generate_structured_refusal():
    """Map provider refusal content to StructuredOutputError."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        status="completed",
        id="resp_refused_structured",
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(type="refusal", refusal="Cannot answer.")]
            )
        ],
        output_parsed=None,
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "refusal"


@pytest.mark.asyncio
async def test_openai_client_generate_structured_incomplete():
    """Map incomplete finish status to StructuredOutputError."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=None,
        status="incomplete",
        id="resp_incomplete",
        output=[],
        output_parsed={"answer": "truncated", "claims": [], "insufficient_evidence": True},
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "incomplete"


@pytest.mark.asyncio
async def test_openai_client_generate_structured_malformed_payload():
    """Fail closed when parsed output violates AnswerV1 invariants."""
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        status="completed",
        id="resp_bad",
        output=[],
        output_parsed={
            "schema_version": "1",
            "answer": "Bad",
            "claims": [{"claim": "No cites", "citation_chunk_ids": []}],
            "insufficient_evidence": False,
        },
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "validation"


@pytest.mark.asyncio
async def test_openai_client_generate_structured_unsupported_without_parse():
    """Fail closed when the injected client has no responses.parse method."""
    responses = SimpleNamespace(
        create=_FakeResponses().create,
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_structured("What is RAG?", AnswerV1)

    assert exc_info.value.category == "unsupported"


@pytest.mark.asyncio
async def test_openai_client_configures_sdk_timeout_without_sdk_retries(monkeypatch):
    """Create the async SDK client with the configured timeout and no SDK retries.

    :param monkeypatch: Pytest fixture used to replace the OpenAI SDK constructor."""
    constructed_clients = []
    responses = _FakeResponses()

    def fake_async_openai(**kwargs):
        """Capture SDK client options and provide a fake Responses endpoint."""
        constructed_clients.append(kwargs)
        return SimpleNamespace(responses=responses)

    openai_module = ModuleType("openai")
    openai_module.AsyncOpenAI = fake_async_openai
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    client = OpenAIResponsesClient(
        "gpt-test", api_key="test-key", timeout_seconds=12.5
    )

    result = await client.generate("What is RAG?")

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

    async def create(self, **kwargs):
        """Record the call and return or raise the next outcome."""
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def parse(self, **kwargs):
        """Record a structured parse call and return or raise the next outcome."""
        return await self.create(**kwargs)


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

    async def record_sleep(seconds: float) -> None:
        """Record the backoff delay without waiting.

        :param seconds: Delay that would have been awaited before the next attempt."""
        delays.append(seconds)

    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        max_retries=max_retries,
        sleeper=record_sleep,
        rng=lambda: 1.0,
    )
    return client, delays


@pytest.mark.asyncio
async def test_openai_client_retries_rate_limit_then_returns_success():
    """Retry HTTP 429 once, then map the successful response."""
    responses = _ScriptedResponses([_StatusFailure(429), _completed_response()])
    client, delays = _retrying_client(responses)

    result = await client.generate("What is RAG?")

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
@pytest.mark.asyncio
async def test_openai_client_retries_transient_and_transport_failures(failure):
    """Retry selected server errors, timeouts, and connection failures.

    :param failure: Retryable error raised by the first provider call."""
    responses = _ScriptedResponses([failure, _completed_response()])
    client, delays = _retrying_client(responses)

    result = await client.generate("What is RAG?")

    assert result.provider_request_id == "resp_ok"
    assert len(responses.calls) == 2
    assert delays == [0.5]


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 501])
@pytest.mark.asyncio
async def test_openai_client_does_not_retry_client_or_unselected_errors(status_code):
    """Leave authentication, bad requests, and unselected statuses unretried.

    :param status_code: Provider status that must fail on the first attempt."""
    responses = _ScriptedResponses([_StatusFailure(status_code), _completed_response()])
    client, delays = _retrying_client(responses)

    with pytest.raises(_StatusFailure, match=f"status {status_code}"):
        await client.generate("What is RAG?")

    assert len(responses.calls) == 1
    assert delays == []


@pytest.mark.asyncio
async def test_openai_client_stops_at_the_retry_limit():
    """Make one initial request plus the configured number of extra attempts."""
    responses = _ScriptedResponses([_StatusFailure(429)] * 4)
    client, delays = _retrying_client(responses, max_retries=2)

    with pytest.raises(_StatusFailure, match="status 429"):
        await client.generate("What is RAG?")

    assert len(responses.calls) == 3
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ConnectionError("reset")])
@pytest.mark.asyncio
async def test_openai_client_retries_transport_failures_at_most_once(failure):
    """Timeouts and connection errors get one retry even when max_retries is higher.

    :param failure: Transport error raised on every provider call."""
    responses = _ScriptedResponses([failure, failure, failure])
    client, delays = _retrying_client(responses, max_retries=2)

    with pytest.raises(type(failure)):
        await client.generate("What is RAG?")

    assert len(responses.calls) == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_openai_client_stops_retrying_when_wall_clock_budget_exceeded():
    """Do not start another attempt after the retry wall-clock budget."""
    responses = _ScriptedResponses([_StatusFailure(503), _completed_response()])
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        """Record the backoff delay without waiting.

        :param seconds: Delay that would have been awaited before the next attempt."""
        delays.append(seconds)

    readings = iter([0.0, 60.0])
    client = OpenAIResponsesClient(
        "gpt-test",
        client=SimpleNamespace(responses=responses),
        max_retries=2,
        sleeper=record_sleep,
        rng=lambda: 1.0,
        clock=lambda: next(readings),
    )

    with pytest.raises(_StatusFailure, match="status 503"):
        await client.generate("What is RAG?")

    assert len(responses.calls) == 1
    assert delays == []


@pytest.mark.asyncio
async def test_openai_client_logs_possible_duplicate_on_transport_retry(caplog):
    """Mark transport retries as possible duplicate generations."""
    responses = _ScriptedResponses([TimeoutError("timed out"), _completed_response()])
    client, _delays = _retrying_client(responses)
    caplog.set_level(logging.WARNING, logger="app.llm.generation")

    result = await client.generate("What is RAG?")

    assert result.provider_request_id == "resp_ok"
    retry_logs = [record for record in caplog.records if record.message == "Retrying generation"]
    assert len(retry_logs) == 1
    assert retry_logs[0].retry_class == "transport"
    assert retry_logs[0].possible_duplicate_generation is True


@pytest.mark.asyncio
async def test_openai_client_makes_one_attempt_when_retries_are_disabled():
    """Do not retry when the retry allowance is zero."""
    responses = _ScriptedResponses([TimeoutError("timed out"), _completed_response()])
    client, delays = _retrying_client(responses, max_retries=0)

    with pytest.raises(TimeoutError, match="timed out"):
        await client.generate("What is RAG?")

    assert len(responses.calls) == 1
    assert delays == []


@pytest.mark.asyncio
async def test_openai_client_caps_backoff_and_omits_the_question_from_retry_logs(caplog):
    """Cap backoff at eight seconds and keep the question out of retry logs.

    :param caplog: Pytest fixture used to capture structured log records."""
    responses = _ScriptedResponses([_StatusFailure(503)] * 7)
    client, delays = _retrying_client(responses, max_retries=6)
    caplog.set_level(logging.WARNING, logger="app.llm.generation")

    with pytest.raises(_StatusFailure):
        await client.generate("private prompt text")

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


@pytest.mark.asyncio
async def test_fake_client_scripts_tool_call_then_structured_answer():
    """Script a tool-call turn followed by a final structured AnswerV1."""
    tools = tool_definitions(search_max_limit=5)
    client = FakeLLMClient(
        tool_turns=[
            ToolTurnResult(
                kind="tool_calls",
                tool_calls=[
                    ToolCallProposal(
                        call_id="call-1",
                        name="search_documents",
                        arguments={"query": "fastapi", "limit": 2},
                    )
                ],
            ),
            ToolTurnResult(
                kind="structured",
                structured=StructuredGenerationResult(
                    parsed=AnswerV1(
                        answer="FastAPI uses Depends.",
                        claims=[
                            AnswerClaim(
                                claim="Depends wires dependencies",
                                citation_chunk_ids=["c1"],
                            )
                        ],
                        insufficient_evidence=False,
                    ),
                    provider="fake",
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                    finish_status="completed",
                ),
            ),
        ]
    )

    first = await client.generate_tool_turn(
        "How does FastAPI inject deps?",
        tools=tools,
        output_model=AnswerV1,
    )
    assert first.kind == "tool_calls"
    assert first.tool_calls is not None
    assert first.tool_calls[0].name == "search_documents"

    second = await client.generate_tool_turn(
        [{"type": "function_call_output", "call_id": "call-1", "output": "[]"}],
        tools=tools,
        output_model=AnswerV1,
    )
    assert second.kind == "structured"
    assert second.structured is not None
    assert second.structured.parsed.answer == "FastAPI uses Depends."


@pytest.mark.asyncio
async def test_fake_client_generate_tool_turn_exhaustion_fails_closed():
    """Fail closed when no scripted tool turns remain."""
    client = FakeLLMClient(tool_turns=[])

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_tool_turn("q", tools=[], output_model=AnswerV1)

    assert exc_info.value.category == "unsupported"


@pytest.mark.asyncio
async def test_openai_client_maps_function_call_output_to_tool_proposals():
    """Map Responses function_call items into ToolCallProposal values."""
    tools = tool_definitions(search_max_limit=5)
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        status="completed",
        id="resp_tools",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_abc",
                name="search_documents",
                arguments='{"query":"budget","limit":3}',
            )
        ],
        output_parsed=None,
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    turn = await client.generate_tool_turn(
        "What is context budgeting?",
        tools=tools,
        output_model=AnswerV1,
    )

    assert responses.calls[0]["tools"] == tools
    assert responses.calls[0]["input"] == "What is context budgeting?"
    assert "text_format" in responses.calls[0]
    assert turn.kind == "tool_calls"
    assert turn.tool_calls == [
        ToolCallProposal(
            call_id="call_abc",
            name="search_documents",
            arguments={"query": "budget", "limit": 3},
        )
    ]


@pytest.mark.asyncio
async def test_openai_client_maps_final_structured_tool_turn():
    """Map a final structured parse during a tool-enabled turn."""
    tools = tool_definitions(search_max_limit=5)
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=8, output_tokens=6),
        status="completed",
        id="resp_final",
        output=[],
        output_parsed=AnswerV1(
            answer="Budgeting reserves output tokens.",
            claims=[
                AnswerClaim(claim="Output reserve exists", citation_chunk_ids=["c2"])
            ],
            insufficient_evidence=False,
        ),
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    turn = await client.generate_tool_turn(
        [
            {"role": "user", "content": "Explain budgeting"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '[{"chunk_id":"c2"}]',
            },
        ],
        tools=tools,
        output_model=AnswerV1,
    )

    assert turn.kind == "structured"
    assert turn.structured is not None
    assert turn.structured.parsed.answer == "Budgeting reserves output tokens."
    assert turn.structured.provider_request_id == "resp_final"


@pytest.mark.asyncio
async def test_openai_client_malformed_tool_arguments_fail_closed():
    """Reject malformed provider tool argument JSON with StructuredOutputError."""
    tools = tool_definitions(search_max_limit=5)
    responses = _FakeResponses()
    responses.response = SimpleNamespace(
        output_text="",
        model="gpt-test-2026",
        usage=None,
        status="completed",
        id="resp_bad_args",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_bad",
                name="search_documents",
                arguments="{not-json",
            )
        ],
        output_parsed=None,
    )
    client = OpenAIResponsesClient(
        "gpt-test", client=SimpleNamespace(responses=responses)
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.generate_tool_turn("q", tools=tools, output_model=AnswerV1)

    assert exc_info.value.category == "validation"
