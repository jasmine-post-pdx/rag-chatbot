"""LLM, embedding, and related model adapters.

Generation clients expose async ``generate``, ``generate_structured``, and
``generate_tool_turn`` methods that callers must await.
"""

from app.llm.generation import (
    FakeLLMClient,
    GenerationResult,
    LLMClient,
    MissingAPIKeyError,
    OpenAIResponsesClient,
    StructuredGenerationResult,
    ToolTurnResult,
)

__all__ = [
    "FakeLLMClient",
    "GenerationResult",
    "LLMClient",
    "MissingAPIKeyError",
    "OpenAIResponsesClient",
    "StructuredGenerationResult",
    "ToolTurnResult",
]
