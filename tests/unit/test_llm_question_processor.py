"""Unit tests for the LLM-backed question processor."""

import logging

import pytest

from app.domain.errors import ContextBudgetExceededError, StructuredOutputError
from app.domain.model_outputs import AnswerClaim, AnswerV1
from app.domain.questions import ModelMetrics, QuestionResult, RetrievalMetrics
from app.llm import FakeLLMClient, StructuredGenerationResult
from app.services.question_processor import (
    LLMQuestionProcessor,
    PlaceholderQuestionProcessor,
    QuestionProcessor,
)
from app.services.question_workflow import QuestionProcessingError, QuestionWorkflow
from app.services.token_budget import ApproximateTokenCounter, ContextBudgetConfig


def _answer(
    text: str = "A generated answer",
    *,
    insufficient_evidence: bool = True,
    claims: list[AnswerClaim] | None = None,
) -> AnswerV1:
    """Build a valid AnswerV1 for processor fakes.

    :param text: Answer text to embed.
    :param insufficient_evidence: Abstention flag.
    :param claims: Optional claims; defaults to empty when abstaining.
    :returns: A validated AnswerV1 instance."""
    return AnswerV1(
        answer=text,
        claims=[] if claims is None else claims,
        insufficient_evidence=insufficient_evidence,
    )


class RecordingClient:
    """Return a fixed structured result and record prompts received.

    :ivar prompts: Prompts passed to ``generate_structured``, in call order."""

    def __init__(self, parsed: AnswerV1 | None = None) -> None:
        """Initialize the client with no recorded prompts.

        :param parsed: Optional structured answer; defaults to an abstention."""
        self.prompts: list[str] = []
        self._parsed = parsed if parsed is not None else _answer()

    async def generate_structured(self, prompt: str, output_model: type):
        """Record the prompt and return fixed structured usage metadata.

        :param prompt: Prompt text to record.
        :param output_model: Expected output model type.
        :returns: A fixed structured answer with known provider metadata."""
        assert output_model is AnswerV1
        self.prompts.append(prompt)
        return StructuredGenerationResult(
            parsed=self._parsed,
            provider="test-provider",
            model="test-model",
            input_tokens=12,
            output_tokens=34,
            finish_status="completed",
            provider_request_id="resp_ok",
        )


class FailingClient:
    """Simulate a provider that fails every structured generation request."""

    async def generate_structured(self, prompt: str, output_model: type):
        """Raise a provider failure.

        :param prompt: Prompt that would have been answered.
        :param output_model: Expected output model type.
        :raises RuntimeError: Always, to simulate an unavailable provider."""
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_processor_maps_structured_result_to_question_result():
    """Map structured answer text and provider usage onto the question result."""
    client = RecordingClient(
        parsed=_answer(
            "A generated answer",
            insufficient_evidence=False,
            claims=[AnswerClaim(claim="Claim", citation_chunk_ids=["c1"])],
        )
    )
    processor = LLMQuestionProcessor(client)

    result = await processor.process("req-1", "What is RAG?")

    assert client.prompts == ["What is RAG?"]
    # Structured claims stay internal; the public result still has empty citations
    # until retrieval grounding is wired.
    assert result == QuestionResult(
        question_id="req-1",
        answer="A generated answer",
        citations=[],
        retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
        model=ModelMetrics(
            provider="test-provider",
            name="test-model",
            input_tokens=12,
            output_tokens=34,
        ),
    )


@pytest.mark.asyncio
async def test_processor_logs_generation_metadata_with_correlation_and_question_ids(caplog):
    """Log provider metadata, usage, and both IDs after successful generation.

    :param caplog: Pytest fixture used to capture structured log records."""
    caplog.set_level(logging.INFO, logger="app.services.question_processor")
    processor = LLMQuestionProcessor(RecordingClient())

    await processor.process(
        "question-job-1",
        "What is RAG?",
        request_id="http-request-1",
    )

    record = next(
        record
        for record in caplog.records
        if record.name == "app.services.question_processor"
        and record.message == "Generation completed"
    )
    assert record.request_id == "http-request-1"
    assert record.question_id == "question-job-1"
    assert record.provider == "test-provider"
    assert record.model == "test-model"
    assert record.elapsed_ms >= 0
    assert record.input_tokens == 12
    assert record.output_tokens == 34
    assert record.insufficient_evidence is True
    assert record.claim_count == 0


@pytest.mark.asyncio
async def test_processor_maps_missing_usage_without_logging_answer_text(caplog):
    """Preserve null usage and omit answer or claim text from success logs.

    :param caplog: Pytest fixture used to capture structured log records."""

    class MissingUsageClient:
        """Return structured success with no reported token usage."""

        provider = "test-provider"
        model = "test-model"

        async def generate_structured(self, prompt: str, output_model: type):
            """Return a result whose provider omitted token usage."""
            return StructuredGenerationResult(
                parsed=_answer("Cannot answer that."),
                provider="test-provider",
                model="test-model",
                input_tokens=None,
                output_tokens=None,
                finish_status="completed",
                provider_request_id="resp_ok",
            )

    caplog.set_level(logging.INFO, logger="app.services.question_processor")
    result = await LLMQuestionProcessor(MissingUsageClient()).process(
        "question-job-3",
        "private prompt text",
        request_id="http-request-3",
    )

    assert result.model == ModelMetrics(
        provider="test-provider",
        name="test-model",
        input_tokens=None,
        output_tokens=None,
    )
    record = next(
        record
        for record in caplog.records
        if record.message == "Generation completed"
    )
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.finish_status == "completed"
    assert record.provider_request_id == "resp_ok"
    assert "private prompt text" not in record.message
    assert "Cannot answer that." not in record.message
    formatted = logging.Formatter().format(record)
    assert "Cannot answer that." not in formatted


@pytest.mark.asyncio
async def test_processor_logs_sanitized_failure_category_without_question(caplog):
    """Log a safe categorized provider failure without exposing prompt text.

    :param caplog: Pytest fixture used to capture structured log records."""

    class FailingPromptClient:
        """Raise a provider-style error containing sensitive request values."""

        provider = "test-provider"
        model = "test-model"

        async def generate_structured(self, prompt: str, output_model: type):
            """Raise an error that echoes the prompt and a credential pattern."""
            raise RuntimeError(f"Rejected question: {prompt}; password=secret")

    caplog.set_level(logging.ERROR, logger="app.services.question_processor")
    processor = LLMQuestionProcessor(FailingPromptClient())

    with pytest.raises(RuntimeError):
        await processor.process(
            "question-job-2",
            "private prompt text",
            request_id="http-request-2",
        )

    record = next(
        record
        for record in caplog.records
        if record.name == "app.services.question_processor"
        and record.message.startswith("Generation failed:")
    )
    assert record.request_id == "http-request-2"
    assert record.question_id == "question-job-2"
    assert record.provider == "test-provider"
    assert record.model == "test-model"
    assert record.elapsed_ms >= 0
    assert record.error_category == "provider_error"
    assert record.exception_type == "RuntimeError"
    assert "private prompt text" not in record.message
    assert "secret" not in record.message


@pytest.mark.asyncio
async def test_processor_returns_empty_citations_and_zero_retrieval():
    """Verify non-RAG answers report no citations and no retrieval work."""
    result = await LLMQuestionProcessor(FakeLLMClient()).process("req-1", "What is RAG?")

    assert result.answer == "Fake answer to: What is RAG?"
    assert result.citations == []
    assert result.retrieval == RetrievalMetrics(
        documents_considered=0, chunks_returned=0
    )


@pytest.mark.asyncio
async def test_processor_propagates_client_errors():
    """Verify client failures are not swallowed so the workflow can record them."""
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await LLMQuestionProcessor(FailingClient()).process("req-1", "What is RAG?")


@pytest.mark.asyncio
async def test_processor_raises_on_structured_refusal(caplog):
    """Refuse structured refusals without inventing an answer.

    :param caplog: Pytest fixture used to capture structured log records."""
    client = FakeLLMClient(structured_refusal="Cannot answer that.")
    processor = LLMQuestionProcessor(client)
    caplog.set_level(logging.ERROR, logger="app.services.question_processor")

    with pytest.raises(StructuredOutputError) as raised:
        await processor.process("req-1", "private prompt text", request_id="http-1")

    assert raised.value.category == "refusal"
    record = next(
        item
        for item in caplog.records
        if item.message.startswith("Generation failed:")
    )
    assert record.error_category == "structured_output_refusal"
    assert "private prompt text" not in record.message
    assert "Cannot answer that." not in record.message


@pytest.mark.asyncio
async def test_processor_retries_incomplete_structured_output_once(caplog):
    """Retry one incomplete structured finish, then succeed.

    :param caplog: Pytest fixture used to capture structured log records."""

    class IncompleteThenOkClient:
        """Fail once as incomplete, then return a valid structured answer."""

        provider = "test-provider"
        model = "test-model"
        calls = 0

        async def generate_structured(self, prompt: str, output_model: type):
            """Raise incomplete on the first call, then return success."""
            type(self).calls += 1
            if type(self).calls == 1:
                raise StructuredOutputError("incomplete")
            return StructuredGenerationResult(
                parsed=_answer("Recovered answer"),
                provider="test-provider",
                model="test-model",
                input_tokens=1,
                output_tokens=2,
                finish_status="completed",
            )

    IncompleteThenOkClient.calls = 0
    caplog.set_level(logging.WARNING, logger="app.services.question_processor")
    result = await LLMQuestionProcessor(IncompleteThenOkClient()).process(
        "req-1", "What is RAG?"
    )

    assert IncompleteThenOkClient.calls == 2
    assert result.answer == "Recovered answer"
    assert any(
        record.message == "Retrying incomplete structured output"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_processor_stops_after_bounded_incomplete_retries(caplog):
    """Fail closed when structured output stays incomplete after one retry.

    :param caplog: Pytest fixture used to capture structured log records."""
    client = FakeLLMClient(structured_incomplete=True)
    processor = LLMQuestionProcessor(client)
    caplog.set_level(logging.ERROR, logger="app.services.question_processor")

    with pytest.raises(StructuredOutputError) as raised:
        await processor.process("req-1", "What is RAG?")

    assert raised.value.category == "incomplete"
    record = next(
        item
        for item in caplog.records
        if item.message.startswith("Generation failed:")
    )
    assert record.error_category == "structured_output_incomplete"


@pytest.mark.asyncio
async def test_processor_raises_on_structured_validation_failure():
    """Fail closed when the structured payload violates AnswerV1 invariants."""
    client = FakeLLMClient(
        structured_payload={
            "schema_version": "1",
            "answer": "Bad",
            "claims": [{"claim": "No cites", "citation_chunk_ids": []}],
            "insufficient_evidence": False,
        }
    )

    with pytest.raises(StructuredOutputError) as raised:
        await LLMQuestionProcessor(client).process("req-1", "What is RAG?")

    assert raised.value.category == "validation"


@pytest.mark.asyncio
async def test_processor_satisfies_question_processor_interface(store_factory):
    """Verify the processor plugs into the workflow as a question processor."""
    processor: QuestionProcessor = LLMQuestionProcessor(FakeLLMClient(model="fake-model"))
    workflow = QuestionWorkflow(
        store_factory(":memory:"), processor, question_id_factory=lambda: "req-1"
    )

    result = await workflow.submit_question("What is RAG?")

    assert result.answer == "Fake answer to: What is RAG?"
    assert result.model.provider == "fake"
    assert result.model.name == "fake-model"
    job = await workflow.get_job("req-1")
    assert job is not None
    assert job.status == "completed"
    assert job.result == result


@pytest.mark.parametrize(
    ("error_type", "expected_category"),
    [
        (type("APITimeoutError", (TimeoutError,), {}), "provider_timeout"),
        (type("RequestTimeout", (Exception,), {}), "provider_timeout"),
        (type("RateLimitError", (Exception,), {}), "provider_rate_limit"),
        (type("Rate_LimitError", (Exception,), {}), "provider_rate_limit"),
        (type("AuthenticationError", (Exception,), {}), "provider_authentication"),
        (type("PermissionDeniedError", (Exception,), {}), "provider_authentication"),
        (type("BadRequestError", (Exception,), {}), "provider_error"),
    ],
    ids=[
        "api-timeout",
        "timeout-name",
        "rate-limit",
        "rate-limit-underscore",
        "authentication",
        "permission",
        "bad-request",
    ],
)
@pytest.mark.asyncio
async def test_workflow_logs_failure_class_and_persists_processing_error(
    store_factory, caplog, error_type, expected_category
):
    """Log the provider failure class and store the public processing error.

    :param store_factory: Fixture that creates a SQLite job store.
    :param caplog: Pytest fixture used to capture structured log records.
    :param error_type: Exception class raised by the fake provider client.
    :param expected_category: Log category expected for that exception class."""

    class ClassifiedFailureClient:
        """Raise one named provider failure without revealing the question."""

        provider = "openai"
        model = "gpt-test"

        async def generate_structured(self, prompt: str, output_model: type):
            """Raise the parametrized provider error, including the prompt."""
            raise error_type(f"Rejected question: {prompt}")

    caplog.set_level(logging.ERROR, logger="app.services.question_processor")
    workflow = QuestionWorkflow(
        store_factory(":memory:"),
        LLMQuestionProcessor(ClassifiedFailureClient()),
        question_id_factory=lambda: "req-1",
    )

    with pytest.raises(QuestionProcessingError) as raised:
        await workflow.submit_question("private prompt text", request_id="http-request-1")

    assert raised.value.exception_type == error_type.__name__
    job = await workflow.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
    record = next(
        record
        for record in caplog.records
        if record.name == "app.services.question_processor"
        and record.message.startswith("Generation failed:")
    )
    assert record.error_category == expected_category
    assert record.exception_type == error_type.__name__
    assert record.request_id == "http-request-1"
    assert record.question_id == "req-1"
    assert "private prompt text" not in record.message


@pytest.mark.asyncio
async def test_workflow_marks_job_failed_when_processor_client_fails(store_factory):
    """Verify client failures through the processor are persisted as failed jobs."""
    workflow = QuestionWorkflow(
        store_factory(":memory:"),
        LLMQuestionProcessor(FailingClient()),
        question_id_factory=lambda: "req-1",
    )

    with pytest.raises(QuestionProcessingError):
        await workflow.submit_question("What is RAG?")

    job = await workflow.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"


@pytest.mark.asyncio
async def test_workflow_persists_structured_output_failure(store_factory):
    """Persist structured refusals as failed jobs with the public error category."""
    workflow = QuestionWorkflow(
        store_factory(":memory:"),
        LLMQuestionProcessor(FakeLLMClient(structured_refusal="nope")),
        question_id_factory=lambda: "req-structured",
    )

    with pytest.raises(QuestionProcessingError):
        await workflow.submit_question("What is RAG?")

    job = await workflow.get_job("req-structured")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"


def _tiny_budget(*, instructions: str | None = None) -> ContextBudgetConfig:
    """Build a budget that only fits a very short question.

    :param instructions: Optional instructions counted as fixed input.
    :returns: A context-budget configuration with a tiny window."""
    return ContextBudgetConfig(
        counter=ApproximateTokenCounter(),
        context_window_tokens=8,
        output_reserve_tokens=4,
        safety_margin_tokens=0,
        instructions=instructions,
        debug=True,
    )


@pytest.mark.asyncio
async def test_processor_with_budget_sends_assembled_prompt_for_empty_chunks():
    """Budget the empty-chunk path and still send the bare question to generate."""
    client = RecordingClient()
    processor = LLMQuestionProcessor(
        client,
        budget=ContextBudgetConfig(
            counter=ApproximateTokenCounter(),
            context_window_tokens=128000,
            output_reserve_tokens=1024,
            safety_margin_tokens=256,
        ),
    )

    result = await processor.process("req-1", "What is RAG?", request_id="http-1")

    assert client.prompts == ["What is RAG?"]
    assert result.answer == "A generated answer"


@pytest.mark.asyncio
async def test_processor_rejects_over_budget_fixed_input_before_generate(caplog):
    """Fail closed on oversized fixed input without calling the provider.

    :param caplog: Pytest fixture used to capture structured log records."""
    client = RecordingClient()
    processor = LLMQuestionProcessor(client, budget=_tiny_budget())
    caplog.set_level(logging.ERROR, logger="app.services.question_processor")

    with pytest.raises(ContextBudgetExceededError):
        await processor.process(
            "req-over",
            "This question is definitely too long for the tiny window",
            request_id="http-over",
        )

    assert client.prompts == []
    record = next(
        item
        for item in caplog.records
        if item.name == "app.services.question_processor"
        and item.message.startswith("Generation failed:")
    )
    assert record.error_category == "context_budget_exceeded"
    assert "This question is definitely too long" not in record.message


@pytest.mark.asyncio
async def test_placeholder_processor_rejects_over_budget_fixed_input():
    """Local placeholder path also enforces the configured context budget."""
    processor = PlaceholderQuestionProcessor(budget=_tiny_budget())

    with pytest.raises(ContextBudgetExceededError):
        await processor.process(
            "req-over",
            "This question is definitely too long for the tiny window",
        )


@pytest.mark.asyncio
async def test_workflow_persists_context_budget_failure(store_factory):
    """Persist over-budget failures as processing errors without provider calls."""
    client = RecordingClient()
    workflow = QuestionWorkflow(
        store_factory(":memory:"),
        LLMQuestionProcessor(client, budget=_tiny_budget()),
        question_id_factory=lambda: "req-budget",
    )

    with pytest.raises(QuestionProcessingError):
        await workflow.submit_question(
            "This question is definitely too long for the tiny window"
        )

    assert client.prompts == []
    job = await workflow.get_job("req-budget")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
