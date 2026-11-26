"""Test double DocumentStore with token-overlap search for grounded tests.

``InMemoryDocumentStore`` requires the full query to be a substring of chunk
text, which is too brittle for realistic question strings. This overlap store
keeps the same protocol while matching when any significant query token appears
in a chunk.
"""

from __future__ import annotations

import re

from app.retrieval.document_store import DocumentChunk, SearchHit

_DEFAULT_MAX_LIMIT = 20


def _tokens(text: str) -> set[str]:
    """Return significant lowercase tokens from ``text``.

    :param text: Free text to tokenize.
    :returns: Tokens of length >= 4."""
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) >= 4}


class TokenOverlapDocumentStore:
    """Search chunks by overlapping significant tokens; look up by chunk ID.

    :ivar _chunks: Indexed chunks keyed by ``chunk_id``.
    :ivar _order: Insertion order used for stable ranking.
    :ivar _max_limit: Hard maximum on ``search_documents`` result size.
    :ivar search_calls: Keyword args from each ``search_documents`` call."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        max_limit: int = _DEFAULT_MAX_LIMIT,
    ) -> None:
        """Index the provided chunks.

        :param chunks: Chunks to make searchable.
        :param max_limit: Hard maximum results returned by search.
        :raises ValueError: If ``max_limit`` is less than 1 or chunk IDs collide."""
        if max_limit < 1:
            raise ValueError("max_limit must be >= 1")
        indexed: dict[str, DocumentChunk] = {}
        order: list[str] = []
        for chunk in chunks:
            if chunk.chunk_id in indexed:
                raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id}")
            indexed[chunk.chunk_id] = chunk
            order.append(chunk.chunk_id)
        self._chunks = indexed
        self._order = order
        self._max_limit = max_limit
        self.search_calls: list[dict] = []

    def search_documents(
        self,
        query: str,
        *,
        framework: str | None = None,
        language: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Return chunks that share significant tokens with ``query``.

        :param query: Free-text query.
        :param framework: Optional framework filter (case-insensitive).
        :param language: Optional language filter (case-insensitive).
        :param limit: Requested maximum hits; capped by the store hard max.
        :returns: Matching hits in insertion order, possibly empty."""
        self.search_calls.append(
            {
                "query": query,
                "framework": framework,
                "language": language,
                "limit": limit,
            }
        )
        effective_limit = min(max(limit, 0), self._max_limit)
        query_tokens = _tokens(query)
        if effective_limit == 0 or not query_tokens:
            return []

        framework_key = framework.casefold() if framework else None
        language_key = language.casefold() if language else None
        hits: list[SearchHit] = []

        for chunk_id in self._order:
            chunk = self._chunks[chunk_id]
            if framework_key is not None and (
                chunk.framework is None or chunk.framework.casefold() != framework_key
            ):
                continue
            if language_key is not None and (
                chunk.language is None or chunk.language.casefold() != language_key
            ):
                continue
            if not query_tokens.intersection(_tokens(chunk.text)):
                continue
            hits.append(
                SearchHit(
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= effective_limit:
                break
        return hits

    def get_document_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Load a single chunk by identifier.

        :param chunk_id: Stable chunk identifier.
        :returns: The chunk, or ``None`` when it does not exist."""
        return self._chunks.get(chunk_id)
