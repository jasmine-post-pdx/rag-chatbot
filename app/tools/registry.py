"""Allow-listed read-only document tool definitions."""

import json
from typing import Any

ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(
    {"search_documents", "get_document_chunk"}
)


def tool_definitions(*, search_max_limit: int = 5) -> list[dict[str, Any]]:
    """Return OpenAI-style function tool definitions for the allow-list.

    :param search_max_limit: Hard maximum advertised for ``search_documents.limit``.
    :returns: Tool definition dictionaries suitable for provider ``tools``."""
    return [
        {
            "type": "function",
            "name": "search_documents",
            "description": (
                "Search the document corpus with an optional framework and "
                "language filter. Returns matching chunk identifiers and text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query.",
                    },
                    "framework": {
                        "type": "string",
                        "description": "Optional framework filter.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional language filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": search_max_limit,
                        "default": min(5, search_max_limit),
                        "description": "Maximum number of hits to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_document_chunk",
            "description": (
                "Load a single document chunk by its stable chunk_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Stable chunk identifier.",
                    },
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    ]


def tool_schema_strings(*, search_max_limit: int = 5) -> list[str]:
    """Serialize allow-listed tool schemas for context budgeting.

    :param search_max_limit: Hard maximum advertised for search limit.
    :returns: JSON strings counted as fixed input by the token budget."""
    return [
        json.dumps(definition, separators=(",", ":"), sort_keys=True)
        for definition in tool_definitions(search_max_limit=search_max_limit)
    ]
