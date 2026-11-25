"""Deterministic in-memory document store for Step 5 tool tests.

Replaced by a real vector-backed store in Step 6 behind the same
``DocumentStore`` protocol.
"""

from app.retrieval.document_store import DocumentChunk, SearchHit

_DEFAULT_MAX_LIMIT = 5


def sample_corpus() -> list[DocumentChunk]:
    """Return a small non-secret corpus for local development and tests.

    :returns: A deterministic list of document chunks."""
    return [
        DocumentChunk(
            source_id="fastapi-guide",
            chunk_id="fastapi-deps",
            text=(
                "FastAPI dependency injection wires routers to services. "
                "Use Depends to inject shared resources into path operations."
            ),
            framework="fastapi",
            language="python",
        ),
        DocumentChunk(
            source_id="fastapi-guide",
            chunk_id="fastapi-lifespan",
            text=(
                "FastAPI lifespan handlers create shared clients on startup "
                "and close them on shutdown."
            ),
            framework="fastapi",
            language="python",
        ),
        DocumentChunk(
            source_id="rag-notes",
            chunk_id="rag-budget",
            text=(
                "Context budgeting reserves tokens for instructions, the "
                "question, tool schemas, output, and a safety margin before "
                "selecting ranked chunks."
            ),
            framework=None,
            language="python",
        ),
        DocumentChunk(
            source_id="express-guide",
            chunk_id="express-middleware",
            text=(
                "Express middleware runs in order for each request and can "
                "attach request IDs or reject invalid input."
            ),
            framework="express",
            language="typescript",
        ),
    ]


class InMemoryDocumentStore:
    """Search and look up chunks held in process memory.

    Matching is a simple case-insensitive substring check so tests stay
    deterministic without embeddings.

    :ivar _chunks: Indexed chunks keyed by ``chunk_id``.
    :ivar _order: Insertion order used for stable search ranking.
    :ivar _max_limit: Hard maximum on ``search_documents`` result size."""

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

    def search_documents(
        self,
        query: str,
        *,
        framework: str | None = None,
        language: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Return chunks whose text contains the query, with optional filters.

        :param query: Free-text query matched as a case-insensitive substring.
        :param framework: Optional framework filter (case-insensitive).
        :param language: Optional language filter (case-insensitive).
        :param limit: Requested maximum hits; capped by the store hard max.
        :returns: Matching hits in insertion order, possibly empty."""
        effective_limit = min(max(limit, 0), self._max_limit)
        if effective_limit == 0 or not query.strip():
            return []

        needle = query.casefold()
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
            if needle not in chunk.text.casefold():
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
