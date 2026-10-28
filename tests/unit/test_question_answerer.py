"""Unit tests for the LLM-backed question answerer."""

import pytest

from app.models import FakeLLMClient, GenerationResult
from app.models.questions import ModelMetrics, RetrievalMetrics
from app.services.question_answerer import QuestionAnswerer
from app.services.question_processor import QuestionProcessor
from app.services.question_workflow import QuestionProcessingError, QuestionWorkflow


class RecordingClient:
    """Return a fixed generation result and record the question it received.

    Attributes:
        questions: Questions passed to ``generate``, in call order.
    """

    def __init__(self) -> None:
        """Initialize the client with no recorded questions."""
        self.questions: list[str] = []

    def generate(self, question: str) -> GenerationResult:
        """Record the question and return fixed usage metadata.

        Args:
            question: User question to record.

        Returns:
            A fixed answer with known provider, model, and token usage.
        """
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

        Args:
            question: User question that would have been answered.

        Raises:
            RuntimeError: Always, to simulate an unavailable provider.
        """
        raise RuntimeError("provider unavailable")


def test_answerer_maps_generation_result_to_question_result():
    """Verify answer, provider, model, and token usage are mapped onto the result."""
    client = RecordingClient()
    answerer = QuestionAnswerer(client)

    result = answerer.process("req-1", "What is RAG?")

    assert client.questions == ["What is RAG?"]
    assert result.request_id == "req-1"
    assert result.answer == "A generated answer"
    assert result.model == ModelMetrics(
        provider="test-provider",
        name="test-model",
        input_tokens=12,
        output_tokens=34,
    )


def test_answerer_returns_empty_citations_and_zero_retrieval():
    """Verify non-RAG answers report no citations and no retrieval work."""
    result = QuestionAnswerer(FakeLLMClient()).process("req-1", "What is RAG?")

    assert result.citations == []
    assert result.retrieval == RetrievalMetrics(
        documents_considered=0, chunks_returned=0
    )


def test_answerer_propagates_client_errors():
    """Verify client failures are not swallowed so the workflow can record them."""
    with pytest.raises(RuntimeError, match="provider unavailable"):
        QuestionAnswerer(FailingClient()).process("req-1", "What is RAG?")


def test_answerer_satisfies_question_processor_interface(store_factory):
    """Verify the answerer plugs into the workflow as a question processor."""
    processor: QuestionProcessor = QuestionAnswerer(FakeLLMClient(model="fake-model"))
    workflow = QuestionWorkflow(store_factory(":memory:"), processor)

    result = workflow.submit("req-1", "What is RAG?")

    assert result.answer == "Fake answer to: What is RAG?"
    assert result.model.provider == "fake"
    assert result.model.name == "fake-model"
    job = workflow.get_job("req-1")
    assert job is not None
    assert job.status == "completed"
    assert job.result == result


def test_workflow_marks_job_failed_when_answerer_client_fails(store_factory):
    """Verify client failures through the answerer are persisted as failed jobs."""
    workflow = QuestionWorkflow(
        store_factory(":memory:"), QuestionAnswerer(FailingClient())
    )

    with pytest.raises(QuestionProcessingError):
        workflow.submit("req-1", "What is RAG?")

    job = workflow.get_job("req-1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_category == "processing_error"
