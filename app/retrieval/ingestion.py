"""Load, chunk, embed, and upsert documents into a vector store."""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger(__name__)

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


class _SourceOutcome(BaseModel):
    """Internal per-source result used while aggregating a concurrent run."""

    source_id: str
    succeeded: bool = False
    chunks_upserted: int = 0
    deleted_before_upsert: bool = False
    failure: IngestSourceFailure | None = None


class BoundedIngestionPipeline:
    """Async ingestion with a concurrency cap on external embed/upsert calls.

    Each source is processed independently under an ``asyncio.Semaphore`` so
    concurrent ``aembed_documents`` calls never exceed ``concurrency``. Sync
    store and chunking work runs via ``asyncio.to_thread``. Per-source failures
    are recorded without aborting siblings.

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
        """Configure the bounded pipeline.

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
        # Serialize sync store mutations; embeds stay concurrent under the semaphore.
        self._store_lock = asyncio.Lock()

    def _chunk_source(self, source: SourceDocument) -> list:
        """Chunk one source on a worker thread.

        :param source: Source document to split.
        :returns: Chunk list for the source."""
        if self._chunking_config is not None:
            return chunk_document(source, config=self._chunking_config)
        return chunk_document(
            source,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    async def _aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via async client API or thread-offloaded sync embed.

        :param texts: Document texts for one source.
        :returns: One embedding per text."""
        aembed = getattr(self._embedder, "aembed_documents", None)
        if aembed is not None:
            return await aembed(texts)
        return await asyncio.to_thread(self._embedder.embed_documents, texts)

    async def _ingest_one(
        self,
        source: SourceDocument,
        semaphore: asyncio.Semaphore,
    ) -> _SourceOutcome:
        """Ingest a single source under the concurrency semaphore.

        :param source: Source to process.
        :param semaphore: Cap on concurrent embed/upsert work.
        :returns: Per-source outcome."""
        outcome = _SourceOutcome(source_id=source.source_id)
        try:
            async with self._store_lock:
                removed = await asyncio.to_thread(
                    self._store.delete_source, source.source_id
                )
            outcome.deleted_before_upsert = bool(removed)
            chunks = await asyncio.to_thread(self._chunk_source, source)
            if not chunks:
                outcome.succeeded = True
                return outcome

            async with semaphore:
                try:
                    vectors = await self._aembed_documents(
                        [chunk.text for chunk in chunks]
                    )
                except Exception:
                    logger.info(
                        "Bounded ingest embed failed source_id=%s category=embed",
                        source.source_id,
                    )
                    outcome.failure = IngestSourceFailure(
                        source_id=source.source_id, category="embed"
                    )
                    return outcome

                model_name = self._embedder.model
                stamped = [
                    EmbeddedChunk(
                        chunk=chunk.model_copy(
                            update={
                                "metadata": {
                                    **chunk.metadata,
                                    "embedding_model": model_name,
                                }
                            }
                        ),
                        embedding=vector,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]

            try:
                async with self._store_lock:
                    await asyncio.to_thread(self._store.upsert_chunks, stamped)
            except Exception:
                logger.info(
                    "Bounded ingest upsert failed source_id=%s category=upsert",
                    source.source_id,
                )
                outcome.failure = IngestSourceFailure(
                    source_id=source.source_id, category="upsert"
                )
                return outcome

            outcome.succeeded = True
            outcome.chunks_upserted = len(stamped)
            return outcome
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info(
                "Bounded ingest failed source_id=%s category=error",
                source.source_id,
            )
            outcome.failure = IngestSourceFailure(
                source_id=source.source_id, category="error"
            )
            return outcome

    def _aggregate(self, outcomes: list[_SourceOutcome]) -> ConcurrentIngestResult:
        """Fold per-source outcomes into a concurrent ingest summary.

        :param outcomes: Completed source outcomes.
        :returns: Aggregated concurrent result."""
        failures = [o.failure for o in outcomes if o.failure is not None]
        succeeded = sum(1 for o in outcomes if o.succeeded)
        return ConcurrentIngestResult(
            sources_processed=len(outcomes),
            chunks_upserted=sum(o.chunks_upserted for o in outcomes),
            sources_deleted_before_upsert=sum(
                1 for o in outcomes if o.deleted_before_upsert
            ),
            sources_succeeded=succeeded,
            sources_failed=len(failures),
            failures=failures,
        )

    async def ingest_documents(
        self, sources: list[SourceDocument]
    ) -> ConcurrentIngestResult:
        """Ingest sources with a bounded concurrency cap on embed calls.

        :param sources: Sources to chunk, embed, and upsert.
        :returns: Concurrent ingestion summary.
        :raises asyncio.CancelledError: When the caller cancels the ingest task."""
        if not sources:
            return ConcurrentIngestResult()

        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            asyncio.create_task(self._ingest_one(source, semaphore))
            for source in sources
        ]
        try:
            raw = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        outcomes: list[_SourceOutcome] = []
        for source, item in zip(sources, raw, strict=True):
            if isinstance(item, asyncio.CancelledError):
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise item
            if isinstance(item, BaseException):
                logger.info(
                    "Bounded ingest failed source_id=%s category=error",
                    source.source_id,
                )
                outcomes.append(
                    _SourceOutcome(
                        source_id=source.source_id,
                        failure=IngestSourceFailure(
                            source_id=source.source_id, category="error"
                        ),
                    )
                )
                continue
            outcomes.append(item)

        result = self._aggregate(outcomes)
        logger.info(
            "Bounded ingest complete processed=%s succeeded=%s failed=%s chunks=%s",
            result.sources_processed,
            result.sources_succeeded,
            result.sources_failed,
            result.chunks_upserted,
        )
        return result

    async def ingest_directory(self, path: Path) -> ConcurrentIngestResult:
        """Ingest ``.md`` files under ``path`` with a concurrency cap.

        :param path: Directory containing markdown sources.
        :returns: Concurrent ingestion summary.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises NotADirectoryError: If ``path`` is not a directory."""
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_dir():
            raise NotADirectoryError(path)

        def _load() -> list[SourceDocument]:
            return [
                load_markdown_source(file_path)
                for file_path in sorted(path.glob("*.md"))
                if file_path.is_file()
            ]

        sources = await asyncio.to_thread(_load)
        return await self.ingest_documents(sources)
