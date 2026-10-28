"""Question answer processing interfaces and placeholder implementation."""

from typing import Protocol

from app.models.questions import (
    ModelMetrics,
    QuestionResult,
    RetrievalMetrics,
)


class QuestionProcessor(Protocol):
    """Produces an answer result for a question."""

    def process(
        self,
        request_id: str,
        question: str,
        *,
        correlation_id: str | None = None,
    ) -> QuestionResult: ...


class PlaceholderQuestionProcessor:
    """Return the current placeholder answer until RAG adapters exist."""

    def process(
        self,
        request_id: str,
        question: str,
        *,
        correlation_id: str | None = None,
    ) -> QuestionResult:
        return QuestionResult(
            request_id=request_id,
            answer=f"This is a placeholder answer for: {question}",
            citations=[],
            retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
            model=ModelMetrics(
                provider="local-or-api",
                name="stub",
                input_tokens=0,
                output_tokens=0,
            ),
        )
