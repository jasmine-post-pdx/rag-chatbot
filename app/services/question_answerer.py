"""Question processor that answers questions with a language model client."""

from app.models.generation import LLMClient
from app.models.questions import (
    ModelMetrics,
    QuestionResult,
    RetrievalMetrics,
)


class QuestionAnswerer:
    """Answer questions by calling an LLM client directly, without retrieval.

    Citations are empty and retrieval counts are zero until RAG is wired up.

    Attributes:
        _client: Language model client used to generate answers.
    """

    def __init__(self, client: LLMClient) -> None:
        """Initialize the answerer with a generation client.

        Args:
            client: Language model client used to generate answers.
        """
        self._client = client

    def process(self, request_id: str, question: str) -> QuestionResult:
        """Generate an answer and map it to a question result.

        Args:
            request_id: Identifier of the question job being processed.
            question: User question to answer.

        Returns:
            The generated answer with model usage and empty retrieval metadata.
        """
        generation = self._client.generate(question)
        return QuestionResult(
            request_id=request_id,
            answer=generation.answer,
            citations=[],
            retrieval=RetrievalMetrics(documents_considered=0, chunks_returned=0),
            model=ModelMetrics(
                provider=generation.provider,
                name=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
            ),
        )
