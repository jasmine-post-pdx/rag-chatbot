"""LLM, embedding, and related model adapters.

Generation clients expose async ``generate`` and ``generate_structured`` methods
that callers must await.
"""

from app.llm.generation import (
    FakeLLMClient,
    GenerationResult,
    LLMClient,
    MissingAPIKeyError,
    OpenAIResponsesClient,
    StructuredGenerationResult,
)

__all__ = [
    "FakeLLMClient",
    "GenerationResult",
    "LLMClient",
    "MissingAPIKeyError",
    "OpenAIResponsesClient",
    "StructuredGenerationResult",
]
