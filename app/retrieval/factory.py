"""Build embedding clients and vector stores from settings."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import Settings
from app.llm.embeddings import EmbeddingClient, FakeEmbeddingClient
from app.retrieval.ingestion import (
    BoundedIngestionPipeline,
    IngestionPipeline,
    build_chunking_config,
)
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.retrieval.vector_store import VectorDocumentStore

logger = logging.getLogger(__name__)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """Construct the configured embedding client.

    :param settings: Application settings.
    :returns: An embedding client.
    :raises ValueError: If the provider name is unsupported.
    :raises RuntimeError: If a local model is requested without dependencies."""
    provider = settings.embedding_provider.casefold()
    if provider == "fake":
        return FakeEmbeddingClient(
            dimensions=settings.embedding_dimensions,
            model=settings.embedding_model,
        )
    if provider == "local":
        from app.llm.local_embeddings import LocalEmbeddingClient

        return LocalEmbeddingClient(model=settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def build_vector_store(
    settings: Settings,
    embedder: EmbeddingClient,
) -> VectorDocumentStore:
    """Construct the configured vector-backed document store.

    :param settings: Application settings.
    :param embedder: Embedding client used for query vectors.
    :returns: A vector document store.
    :raises ValueError: If the backend name is unsupported."""
    backend = settings.vector_store.casefold()
    max_limit = max(settings.search_max_top_k, settings.tool_search_max_limit)
    if backend == "memory":
        return InMemoryVectorDocumentStore(
            embedder,
            max_limit=max_limit,
            similarity_metric=settings.similarity_metric,
        )
    if backend == "pgvector":
        from app.retrieval.pgvector_store import PgvectorDocumentStore

        return PgvectorDocumentStore(
            settings.postgres_dsn,
            embedder,
            collection=settings.vector_collection,
            max_limit=max_limit,
            dimensions=embedder.dimensions,
            similarity_metric=settings.similarity_metric,
        )
    raise ValueError(f"Unsupported vector store: {settings.vector_store}")


def seed_corpus_if_present(
    settings: Settings,
    store: VectorDocumentStore,
    embedder: EmbeddingClient,
) -> None:
    """Ingest markdown sources from ``corpus_raw_dir`` when the directory exists.

    :param settings: Application settings.
    :param store: Destination vector store.
    :param embedder: Embedding client for document vectors."""
    raw_dir = Path(settings.corpus_raw_dir)
    if not raw_dir.is_dir():
        logger.info("Corpus directory missing; skipping seed path=%s", raw_dir)
        return
    markdown_files = list(raw_dir.glob("*.md"))
    if not markdown_files:
        logger.info("Corpus directory has no markdown; skipping seed path=%s", raw_dir)
        return
    pipeline = IngestionPipeline(
        store,
        embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunking_config=build_chunking_config(
            strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    )
    result = pipeline.ingest_directory(raw_dir)
    logger.info(
        "Seeded vector store sources=%s chunks=%s path=%s",
        result.sources_processed,
        result.chunks_upserted,
        raw_dir,
    )


async def async_seed_corpus_if_present(
    settings: Settings,
    store: VectorDocumentStore,
    embedder: EmbeddingClient,
) -> None:
    """Seed the corpus via bounded concurrent ingestion when sources exist.

    :param settings: Application settings.
    :param store: Destination vector store.
    :param embedder: Embedding client for document vectors."""
    raw_dir = Path(settings.corpus_raw_dir)
    if not raw_dir.is_dir():
        logger.info("Corpus directory missing; skipping seed path=%s", raw_dir)
        return
    markdown_files = list(raw_dir.glob("*.md"))
    if not markdown_files:
        logger.info("Corpus directory has no markdown; skipping seed path=%s", raw_dir)
        return
    pipeline = BoundedIngestionPipeline(
        store,
        embedder,
        concurrency=settings.ingestion_concurrency,
        timeout_seconds=settings.ingestion_timeout_seconds,
        source_timeout_seconds=settings.ingestion_source_timeout_seconds,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunking_config=build_chunking_config(
            strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    )
    result = await pipeline.ingest_directory(raw_dir)
    logger.info(
        "Seeded vector store sources=%s succeeded=%s failed=%s chunks=%s path=%s",
        result.sources_processed,
        result.sources_succeeded,
        result.sources_failed,
        result.chunks_upserted,
        raw_dir,
    )
