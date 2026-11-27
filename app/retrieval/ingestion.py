"""Load, chunk, embed, and upsert documents into a vector store."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from app.llm.embeddings import EmbeddingClient
from app.retrieval.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    SourceDocument,
    chunk_document,
)
from app.retrieval.vector_store import EmbeddedChunk, VectorDocumentStore

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class IngestResult(BaseModel):
    """Summary of an ingestion run.

    :ivar sources_processed: Number of sources accepted for ingestion.
    :ivar chunks_upserted: Number of embedded chunks written.
    :ivar sources_deleted_before_upsert: Sources cleared before re-upsert."""

    sources_processed: int = 0
    chunks_upserted: int = 0
    sources_deleted_before_upsert: int = 0


class IngestSourceFailure(BaseModel):
    """Safe per-source failure recorded during bounded concurrent ingestion.

    :ivar source_id: Identifier of the source that failed.
    :ivar category: Stable failure category (e.g. ``timeout``, ``embed``, ``upsert``)."""

    source_id: str
    category: str


class ConcurrentIngestResult(IngestResult):
    """Summary of a bounded concurrent ingestion run.

    :ivar sources_succeeded: Sources that completed without error.
    :ivar sources_failed: Sources that failed (timeout, embed, upsert, etc.).
    :ivar failures: Safe per-source failure records."""

    sources_succeeded: int = 0
    sources_failed: int = 0
    failures: list[IngestSourceFailure] = []


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Parse optional YAML-like ``key: value`` front matter.

    :param raw: File contents.
    :returns: Metadata map and remaining body text."""
    match = _FRONT_MATTER_RE.match(raw)
    if match is None:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            meta[key] = value
    return meta, match.group(2)


def load_markdown_source(path: Path) -> SourceDocument:
    """Load a markdown file into a ``SourceDocument``.

    The file stem becomes ``source_id``. Optional front matter may set
    ``title``, ``framework``, and ``language``.

    :param path: Path to a ``.md`` file.
    :returns: Parsed source document."""
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)
    title = meta.pop("title", path.stem)
    framework = meta.pop("framework", None) or None
    language = meta.pop("language", None) or None
    metadata = {"path": str(path).replace("\\", "/"), **meta}
    return SourceDocument(
        source_id=path.stem,
        title=title,
        text=body,
        framework=framework,
        language=language,
        metadata=metadata,
    )


def build_chunking_config(
    *,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkingConfig:
    """Build a ``ChunkingConfig`` from settings-style fields.

    :param strategy: Strategy name (case-insensitive).
    :param chunk_size: Max size in strategy units.
    :param chunk_overlap: Overlap in strategy units.
    :returns: A configured ``ChunkingConfig``.
    :raises ValueError: If the strategy name is unsupported."""
    normalized = strategy.strip().casefold()
    try:
        selected = ChunkingStrategy(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported chunking strategy: {strategy}") from exc

    token_counter = None
    if selected == ChunkingStrategy.TOKEN_AWARE:
        from app.services.token_budget import ApproximateTokenCounter

        token_counter = ApproximateTokenCounter()

    return ChunkingConfig(
        strategy=selected,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        token_counter=token_counter,
    )


class IngestionPipeline:
    """Chunk, embed, and upsert sources with delete-before-write re-index.

    Re-ingesting a ``source_id`` deletes that source's existing chunks first so
    changed documents never leave stale vectors searchable.

    :ivar _store: Vector-backed document store.
    :ivar _embedder: Embedding client for document vectors.
    :ivar _chunk_size: Character window size used when ``chunking_config`` is unset.
    :ivar _chunk_overlap: Character overlap used when ``chunking_config`` is unset.
    :ivar _chunking_config: Optional strategy config that overrides legacy size kwargs."""

    def __init__(
        self,
        store: VectorDocumentStore,
        embedder: EmbeddingClient,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        """Configure the pipeline.

        :param store: Destination vector store.
        :param embedder: Embedding client.
        :param chunk_size: Character window size when ``chunking_config`` is unset.
        :param chunk_overlap: Character overlap when ``chunking_config`` is unset.
        :param chunking_config: Optional strategy-aware chunking configuration."""
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunking_config = chunking_config

    def ingest_documents(self, sources: list[SourceDocument]) -> IngestResult:
        """Ingest an in-memory list of sources.

        :param sources: Sources to chunk and upsert.
        :returns: Ingestion summary counts."""
        if not sources:
            return IngestResult()

        deleted_sources = 0
        all_chunks = []
        for source in sources:
            removed = self._store.delete_source(source.source_id)
            if removed:
                deleted_sources += 1
            if self._chunking_config is not None:
                chunks = chunk_document(source, config=self._chunking_config)
            else:
                chunks = chunk_document(
                    source,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                )
            all_chunks.extend(chunks)

        if not all_chunks:
            return IngestResult(
                sources_processed=len(sources),
                chunks_upserted=0,
                sources_deleted_before_upsert=deleted_sources,
            )

        vectors = self._embedder.embed_documents([chunk.text for chunk in all_chunks])
        embedded = [
            EmbeddedChunk(chunk=chunk, embedding=vector)
            for chunk, vector in zip(all_chunks, vectors, strict=True)
        ]
        # Stamp embedding model identity into metadata without mutating callers.
        model_name = self._embedder.model
        stamped = [
            EmbeddedChunk(
                chunk=item.chunk.model_copy(
                    update={
                        "metadata": {
                            **item.chunk.metadata,
                            "embedding_model": model_name,
                        }
                    }
                ),
                embedding=item.embedding,
            )
            for item in embedded
        ]
        self._store.upsert_chunks(stamped)
        return IngestResult(
            sources_processed=len(sources),
            chunks_upserted=len(stamped),
            sources_deleted_before_upsert=deleted_sources,
        )

    def ingest_directory(self, path: Path) -> IngestResult:
        """Ingest every ``.md`` file under ``path`` (non-recursive).

        :param path: Directory containing markdown sources.
        :returns: Ingestion summary counts.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises NotADirectoryError: If ``path`` is not a directory."""
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_dir():
            raise NotADirectoryError(path)
        sources = [
            load_markdown_source(file_path)
            for file_path in sorted(path.glob("*.md"))
            if file_path.is_file()
        ]
        return self.ingest_documents(sources)

    def delete_source(self, source_id: str) -> int:
        """Delete all chunks for a source from the store.

        :param source_id: Source identifier to remove.
        :returns: Number of chunks removed."""
        return self._store.delete_source(source_id)


class BoundedIngestionPipeline:
    """Async ingestion with a concurrency cap on external embed/upsert calls.

    Step 9 stub: the public surface exists so TDD tests can collect and fail red
    before the real bounded concurrent implementation is written.

    :ivar concurrency: Max concurrent external calls (1–10).
    :ivar timeout_seconds: Optional overall job deadline.
    :ivar source_timeout_seconds: Optional per-source deadline."""

    def __init__(
        self,
        store: VectorDocumentStore,
        embedder: EmbeddingClient,
        *,
        concurrency: int = 5,
        timeout_seconds: float | None = None,
        source_timeout_seconds: float | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        """Configure the bounded pipeline (stub).

        :param store: Destination vector store.
        :param embedder: Embedding client (sync or async-capable).
        :param concurrency: Max concurrent external calls; must be 1–10.
        :param timeout_seconds: Optional overall job deadline in seconds.
        :param source_timeout_seconds: Optional per-source deadline in seconds.
        :param chunk_size: Character window size when ``chunking_config`` is unset.
        :param chunk_overlap: Character overlap when ``chunking_config`` is unset.
        :param chunking_config: Optional strategy-aware chunking configuration.
        :raises ValueError: If ``concurrency`` is outside 1–10."""
        if concurrency < 1 or concurrency > 10:
            raise ValueError("concurrency must be between 1 and 10")
        self._store = store
        self._embedder = embedder
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds
        self.source_timeout_seconds = source_timeout_seconds
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunking_config = chunking_config

    async def ingest_documents(
        self, sources: list[SourceDocument]
    ) -> ConcurrentIngestResult:
        """Ingest sources with a bounded concurrency cap (Step 9 stub).

        :param sources: Sources to chunk, embed, and upsert.
        :returns: Concurrent ingestion summary.
        :raises NotImplementedError: Until the bounded path is implemented."""
        raise NotImplementedError(
            "BoundedIngestionPipeline.ingest_documents is not implemented"
        )

    async def ingest_directory(self, path: Path) -> ConcurrentIngestResult:
        """Ingest ``.md`` files under ``path`` with a concurrency cap (Step 9 stub).

        :param path: Directory containing markdown sources.
        :returns: Concurrent ingestion summary.
        :raises NotImplementedError: Until the bounded path is implemented."""
        raise NotImplementedError(
            "BoundedIngestionPipeline.ingest_directory is not implemented"
        )
