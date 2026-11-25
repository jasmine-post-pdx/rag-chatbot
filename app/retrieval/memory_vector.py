"""In-memory cosine vector store implementing DocumentStore + upserts."""

from __future__ import annotations

import math

from app.llm.embeddings import EmbeddingClient
from app.retrieval.document_store import DocumentChunk, SearchHit
from app.retrieval.vector_store import EmbeddedChunk

_DEFAULT_MAX_LIMIT = 5


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity between two equal-width vectors.

    :param left: First vector.
    :param right: Second vector.
    :returns: Cosine similarity in ``[-1, 1]``, or ``0.0`` when a norm is zero.
    :raises ValueError: If the vectors have different lengths."""
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class InMemoryVectorDocumentStore:
    """Cosine top-k store kept in process memory.

    ``search_documents`` embeds the query with the injected client so tools and
    HTTP search share the same ranking path.

    :ivar _embedder: Embedding client used for queries.
    :ivar _chunks: Indexed chunks keyed by ``chunk_id``.
    :ivar _embeddings: Embedding vectors keyed by ``chunk_id``.
    :ivar _max_limit: Hard maximum on search result size.
    :ivar similarity_metric: Documented metric label (``cosine``)."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        *,
        max_limit: int = _DEFAULT_MAX_LIMIT,
        similarity_metric: str = "cosine",
    ) -> None:
        """Create an empty in-memory vector store.

        :param embedder: Client used to embed search queries.
        :param max_limit: Hard maximum results returned by search.
        :param similarity_metric: Metric label exposed to callers.
        :raises ValueError: If ``max_limit`` is less than 1."""
        if max_limit < 1:
            raise ValueError("max_limit must be >= 1")
        if similarity_metric != "cosine":
            raise ValueError("only cosine similarity is supported")
        self._embedder = embedder
        self._chunks: dict[str, DocumentChunk] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._max_limit = max_limit
        self.similarity_metric = similarity_metric

    def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or replace chunks by ``chunk_id``.

        :param chunks: Embedded chunks to store.
        :raises ValueError: If an embedding width mismatches an existing vector."""
        for item in chunks:
            width = len(item.embedding)
            if self._embeddings:
                existing_width = len(next(iter(self._embeddings.values())))
                if width != existing_width:
                    raise ValueError("embedding dimensions must match store")
            self._chunks[item.chunk.chunk_id] = item.chunk
            self._embeddings[item.chunk.chunk_id] = list(item.embedding)

    def delete_source(self, source_id: str) -> int:
        """Remove every chunk belonging to ``source_id``.

        :param source_id: Source identifier to clear.
        :returns: Number of chunks removed."""
        to_remove = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.source_id == source_id
        ]
        for chunk_id in to_remove:
            del self._chunks[chunk_id]
            del self._embeddings[chunk_id]
        return len(to_remove)

    def search_documents(
        self,
        query: str,
        *,
        framework: str | None = None,
        language: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Embed the query and return cosine top-k hits with optional filters.

        :param query: Free-text query.
        :param framework: Optional framework filter (case-insensitive).
        :param language: Optional language filter (case-insensitive).
        :param limit: Requested maximum hits; capped by the store hard max.
        :returns: Matching hits in descending score order, possibly empty."""
        effective_limit = min(max(limit, 0), self._max_limit)
        if effective_limit == 0 or not query.strip() or not self._chunks:
            return []

        try:
            query_vector = self._embedder.embed_query(query)
        except ValueError:
            return []

        framework_key = framework.casefold() if framework else None
        language_key = language.casefold() if language else None
        scored: list[tuple[float, DocumentChunk]] = []

        for chunk_id, chunk in self._chunks.items():
            if framework_key is not None and (
                chunk.framework is None or chunk.framework.casefold() != framework_key
            ):
                continue
            if language_key is not None and (
                chunk.language is None or chunk.language.casefold() != language_key
            ):
                continue
            score = cosine_similarity(query_vector, self._embeddings[chunk_id])
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[SearchHit] = []
        for rank, (score, chunk) in enumerate(scored[:effective_limit], start=1):
            hits.append(
                SearchHit(
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    title=chunk.title,
                    metadata=dict(chunk.metadata),
                    rank=rank,
                    score=score,
                )
            )
        return hits

    def get_document_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Load a single chunk by identifier.

        :param chunk_id: Stable chunk identifier.
        :returns: The chunk, or ``None`` when it does not exist."""
        return self._chunks.get(chunk_id)
