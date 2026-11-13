"""LLM, embedding, and related model adapters.

Generation clients expose an async ``generate`` method that callers must await.
"""

from app.llm.generation import (
    FakeLLMClient,
    GenerationResult,
    LLMClient,
    MissingAPIKeyError,
    OpenAIResponsesClient,
)

__all__ = [
    "FakeLLMClient",
    "GenerationResult",
    "LLMClient",
    "MissingAPIKeyError",
    "OpenAIResponsesClient",
]
