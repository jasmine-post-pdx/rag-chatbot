"""Postgres/pgvector-backed document store adapter."""

from __future__ import annotations

import json
from typing import Any

from app.llm.embeddings import EmbeddingClient
from app.retrieval.document_store import DocumentChunk, SearchHit
from app.retrieval.vector_store import EmbeddedChunk

_DEFAULT_MAX_LIMIT = 5


class PgvectorDocumentStore:
    """Cosine vector store persisted in Postgres with the ``vector`` extension.

    Blocking psycopg calls are intended to run via ``asyncio.to_thread`` from
    async callers (tools and search already do this for store methods).

    :ivar _dsn: Postgres connection string.
    :ivar _embedder: Embedding client for query vectors.
    :ivar _collection: Logical collection name stored on each row.
    :ivar _max_limit: Hard maximum search results.
    :ivar _dimensions: Expected embedding width.
    :ivar similarity_metric: Documented metric label."""

    def __init__(
        self,
        dsn: str,
        embedder: EmbeddingClient,
        *,
        collection: str = "documents",
        max_limit: int = _DEFAULT_MAX_LIMIT,
        dimensions: int | None = None,
        similarity_metric: str = "cosine",
    ) -> None:
        """Create the store and ensure schema exists.

        :param dsn: Postgres DSN.
        :param embedder: Client used to embed search queries.
        :param collection: Logical collection/index name.
        :param max_limit: Hard maximum search results.
        :param dimensions: Expected embedding width; defaults to embedder width.
        :param similarity_metric: Metric label (must be ``cosine``).
        :raises ValueError: If arguments are invalid."""
        if max_limit < 1:
            raise ValueError("max_limit must be >= 1")
        if similarity_metric != "cosine":
            raise ValueError("only cosine similarity is supported")
        self._dsn = dsn
        self._embedder = embedder
        self._collection = collection
        self._max_limit = max_limit
        self._dimensions = dimensions or embedder.dimensions
        self.similarity_metric = similarity_metric
        self._ensure_schema()

    def _connect(self) -> Any:
        """Open a new psycopg connection.

        :returns: An open connection."""
        import psycopg

        return psycopg.connect(self._dsn)

    def _ensure_schema(self) -> None:
        """Create the vector extension and chunks table when missing."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        title TEXT,
                        text TEXT NOT NULL,
                        framework TEXT,
                        language TEXT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        embedding vector(%s) NOT NULL
                    )
                    """,
                    (self._dimensions,),
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_source_idx
                    ON document_chunks (collection, source_id)
                    """
                )
            conn.commit()

    def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or replace chunks by ``chunk_id`` within this collection.

        :param chunks: Embedded chunks to store.
        :raises ValueError: If an embedding width mismatches ``dimensions``."""
        if not chunks:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                for item in chunks:
                    if len(item.embedding) != self._dimensions:
                        raise ValueError("embedding dimensions must match store")
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            chunk_id, source_id, collection, title, text,
                            framework, language, metadata, embedding
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            source_id = EXCLUDED.source_id,
                            collection = EXCLUDED.collection,
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            framework = EXCLUDED.framework,
                            language = EXCLUDED.language,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            item.chunk.chunk_id,
                            item.chunk.source_id,
                            self._collection,
                            item.chunk.title,
                            item.chunk.text,
                            item.chunk.framework,
                            item.chunk.language,
                            json.dumps(item.chunk.metadata),
                            _vector_literal(item.embedding),
                        ),
                    )
            conn.commit()

    def delete_source(self, source_id: str) -> int:
        """Remove every chunk for ``source_id`` in this collection.

        :param source_id: Source identifier to clear.
        :returns: Number of chunks removed."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE collection = %s AND source_id = %s
                    """,
                    (self._collection, source_id),
                )
                removed = cur.rowcount
            conn.commit()
        return int(removed)

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
        if effective_limit == 0 or not query.strip():
            return []
        try:
            query_vector = self._embedder.embed_query(query)
        except ValueError:
            return []

        query_literal = _vector_literal(query_vector)
        clauses = ["collection = %s"]
        params: list[Any] = [self._collection]
        if framework is not None:
            clauses.append("LOWER(framework) = LOWER(%s)")
            params.append(framework)
        if language is not None:
            clauses.append("LOWER(language) = LOWER(%s)")
            params.append(language)
        where_sql = " AND ".join(clauses)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                    SELECT chunk_id, source_id, title, text, framework, language,
                           metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM document_chunks
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                (query_literal, *params, query_literal, effective_limit),
            )
            rows = cur.fetchall()

        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            (
                chunk_id,
                source_id,
                title,
                text,
                _framework,
                _language,
                metadata,
                score,
            ) = row
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hits.append(
                SearchHit(
                    source_id=source_id,
                    chunk_id=chunk_id,
                    text=text,
                    title=title,
                    metadata=dict(metadata or {}),
                    rank=rank,
                    score=float(score),
                )
            )
        return hits

    def get_document_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Load a single chunk by identifier within this collection.

        :param chunk_id: Stable chunk identifier.
        :returns: The chunk, or ``None`` when it does not exist."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT source_id, chunk_id, title, text, framework, language,
                           metadata
                    FROM document_chunks
                    WHERE collection = %s AND chunk_id = %s
                    """,
                (self._collection, chunk_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        source_id, chunk_id, title, text, framework, language, metadata = row
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return DocumentChunk(
            source_id=source_id,
            chunk_id=chunk_id,
            text=text,
            title=title,
            framework=framework,
            language=language,
            metadata=dict(metadata or {}),
        )

    def is_ready(self) -> bool:
        """Return whether a simple connectivity probe succeeds.

        :returns: ``True`` when the database accepts a query."""
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:  # noqa: BLE001 - readiness probes must not raise
            return False


def _vector_literal(values: list[float]) -> str:
    """Format a Python float list as a pgvector literal.

    :param values: Embedding values.
    :returns: Literal such as ``[0.1,0.2]``."""
    return "[" + ",".join(str(float(value)) for value in values) + "]"


__all__ = ["PgvectorDocumentStore"]
