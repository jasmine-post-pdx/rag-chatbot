"""Unit tests for the LLM-backed question processor."""

import logging

import pytest

from app.domain.questions import ModelMetrics, RetrievalMetrics
from app.llm import FakeLLMClient, GenerationResult
from app.services.question_processor import LLMQuestionProcessor, QuestionProcessor
from app.services.question_workflow import QuestionProcessingError, QuestionWorkflow


class RecordingClient:
    """Return a fixed generation result and record the question it received.

    :ivar questions: Questions passed to ``generate``, in call order."""

    def __init__(self) -> None:
        """Initialize the client with no recorded questions."""
        self.questions: list[str] = []

    def generate(self, question: str) -> GenerationResult:
        """Record the question and return fixed usage metadata.

        :param question: User question to record.
        :returns: A fixed answer with known provider, model, and token usage."""
        self.questions.append(question)
        return GenerationResult(
            answer="A generated answer",
            provider="test-provider",
            model="test-model",
            input_tokens=12,
            output_tokens=34,
        )


class FailingClient:
    """Simulate a provider that fails every generation request."""

    def generate(self, question: str) -> GenerationResult:
        """Raise a provider failure.

        :param question: User question that would have been answered.
        :raises RuntimeError: Always, to simulate an unavailable provider."""
        raise RuntimeError("provider unavailable")


def test_processor_maps_generation_result_to_question_result():
    """Verify answer, provider, model, and token usage are mapped onto the result."""
    client = RecordingClient()
    processor = LLMQuestionProcessor(client)

    result = processor.process("req-1", "What is RAG?")

    assert client.questions == ["What is RAG?"]
    assert result.question_id == "req-1"
    assert result.answer == "A generated answer"
    assert result.model == ModelMetrics(
        provider="test-provider",
        name="test-model",
        input_tokens=12,
        output_tokens=34,
    )


def test_processor_logs_generation_metadata_with_correlation_and_question_ids(caplog):
    """Log provider metadata, usage, and both IDs after successful generation.

    :param caplog: Pytest fixture used to capture structured log records."""
    caplog.set_level(logging.INFO, logger="app.services.question_processor")
    processor = LLMQuestionProcessor(RecordingClient())

    processor.process(
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


def test_processor_maps_missing_usage_and_logs_status_without_refusal(caplog):
    """Preserve null usage and log status and provider ID, not refusal text.

    :param caplog: Pytest fixture used to capture structured log records."""

    class MissingUsageClient:
        """Return a refusal with no reported token usage."""

        provider = "test-provider"
        model = "test-model"

        def generate(self, question: str) -> GenerationResult:
            """Return a result whose provider omitted token usage."""
            return GenerationResult(
                answer="Cannot answer that.",
                provider="test-provider",
                model="test-model",
                input_tokens=None,
                output_tokens=None,
                finish_status="incomplete",
                refusal="Cannot answer that.",
                provider_request_id="resp_refused",
            )

    caplog.set_level(logging.INFO, logger="app.services.question_processor")
    result = LLMQuestionProcessor(MissingUsageClient()).process(
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
    assert record.finish_status == "incomplete"
    assert record.provider_request_id == "resp_refused"
    assert "private prompt text" not in record.message
    assert "Cannot answer that." not in record.message
    formatted = logging.Formatter().format(record)
    assert "Cannot answer that." not in formatted


def test_processor_logs_sanitized_failure_category_without_question(caplog):
    """Log a safe categorized provider failure without exposing prompt text.

    :param caplog: Pytest fixture used to capture structured log records."""
    class FailingClient:
        """Raise a provider-style error containing sensitive request values."""

        provider = "test-provider"
        model = "test-model"

        def generate(self, question: str) -> GenerationResult:
            """Raise an error that echoes the prompt and a credential pattern."""
            raise RuntimeError(f"Rejected question: {question}; password=secret")

    caplog.set_level(logging.ERROR, logger="app.services.question_processor")
    processor = LLMQuestionProcessor(FailingClient())

    with pytest.raises(RuntimeError):
        processor.process(
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


def test_processor_returns_empty_citations_and_zero_retrieval():
    """Verify non-RAG answers report no citations and no retrieval work."""
    result = LLMQuestionProcessor(FakeLLMClient()).process("req-1", "What is RAG?")

    assert result.citations == []
    assert result.retrieval == RetrievalMetrics(
        documents_considered=0, chunks_returned=0
    )


def test_processor_propagates_client_errors():
    """Verify client failures are not swallowed so the workflow can record them."""
    with pytest.raises(RuntimeError, match="provider unavailable"):
        LLMQuestionProcessor(FailingClient()).process("req-1", "What is RAG?")


def test_processor_satisfies_question_processor_interface(store_factory):
    """Verify the processor plugs into the workflow as a question processor."""
    processor: QuestionProcessor = LLMQuestionProcessor(FakeLLMClient(model="fake-model"))
    workflow = QuestionWorkflow(
        store_factory(":memory:"), processor, question_id_factory=lambda: "req-1"
    )

    result = workflow.submit_question("What is RAG?")

    assert result.answer == "Fake answer to: What is RAG?"
    assert result.model.provider == "fake"
    assert result.model.name == "fake-model"
    job = workflow.get_job("req-1")
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
def test_workflow_logs_failure_class_and_persists_processing_error(
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

        def generate(self, question: str) -> GenerationResult:
            """Raise the parametrized provider error, including the prompt."""
            raise error_type(f"Rejected question: {question}")

    caplog.set_level(logging.ERROR, logger="app.services.question_processor")
    workflow = QuestionWorkflow(
        store_factory(":memory:"),
        LLMQuestionProcessor(ClassifiedFailureClient()),
        question_id_factory=lambda: "req-1",
    )

    with pytest.raises(QuestionProcessingError) as raised:
        workflow.submit_question("private prompt text", request_id="http-request-1")

    assert raised.value.exception_type == error_type.__name__
    job = workflow.get_job("req-1")
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


def test_workflow_marks_job_failed_when_processor_client_fails(store_factory):
    """Verify client failures through the processor are persisted as failed jobs."""
    workflow = QuestionWorkflow(
        store_factory(":memory:"),
        LLMQuestionProcessor(FailingClient()),
        question_id_factory=lambda: "req-1",
    )

    with pytest.raises(QuestionProcessingError):
        workflow.submit_question("What is RAG?")

    job = workflow.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
