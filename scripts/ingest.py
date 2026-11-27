"""CLI to ingest markdown sources into the configured vector store."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config.settings import settings
from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.ingestion import BoundedIngestionPipeline, build_chunking_config
from app.retrieval.memory_vector import InMemoryVectorDocumentStore

logger = logging.getLogger(__name__)


def build_embedder():
    """Build the embedding client from settings.

    :returns: An embedding client instance."""
    provider = settings.embedding_provider.casefold()
    if provider == "fake":
        return FakeEmbeddingClient(
            dimensions=settings.embedding_dimensions,
            model=settings.embedding_model,
        )
    if provider == "local":
        from app.llm.local_embeddings import LocalEmbeddingClient

        return LocalEmbeddingClient(model=settings.embedding_model)
    raise SystemExit(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def build_store(embedder):
    """Build the vector store from settings.

    :param embedder: Embedding client for query-time search.
    :returns: A vector document store."""
    backend = settings.vector_store.casefold()
    if backend == "memory":
        return InMemoryVectorDocumentStore(
            embedder,
            max_limit=settings.search_max_top_k,
            similarity_metric=settings.similarity_metric,
        )
    if backend == "pgvector":
        from app.retrieval.pgvector_store import PgvectorDocumentStore

        return PgvectorDocumentStore(
            settings.postgres_dsn,
            embedder,
            collection=settings.vector_collection,
            max_limit=settings.search_max_top_k,
            dimensions=embedder.dimensions,
            similarity_metric=settings.similarity_metric,
        )
    raise SystemExit(f"Unsupported VECTOR_STORE: {settings.vector_store}")


async def _run_ingest(path: Path) -> int:
    """Run bounded concurrent ingestion for ``path``.

    :param path: Directory of markdown sources.
    :returns: Process exit code."""
    embedder = build_embedder()
    store = build_store(embedder)
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
    try:
        result = await pipeline.ingest_directory(path)
        logger.info(
            "Ingestion complete sources=%s succeeded=%s failed=%s chunks=%s "
            "deleted_sources=%s store=%s model=%s",
            result.sources_processed,
            result.sources_succeeded,
            result.sources_failed,
            result.chunks_upserted,
            result.sources_deleted_before_upsert,
            settings.vector_store,
            embedder.model,
        )
        if settings.vector_store == "memory":
            logger.warning(
                "VECTOR_STORE=memory: ingested vectors exist only in this process "
                "and are not available to a separately started API server"
            )
        return 0
    finally:
        aclose = getattr(embedder, "aclose", None)
        if aclose is not None:
            await aclose()


def main(argv: list[str] | None = None) -> int:
    """Ingest ``data/raw`` (or ``--path``) into the configured store.

    :param argv: Optional CLI argument list.
    :returns: Process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(settings.corpus_raw_dir),
        help="Directory of markdown sources (default: CORPUS_RAW_DIR)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=settings.log_level)
    return asyncio.run(_run_ingest(args.path))


if __name__ == "__main__":
    sys.exit(main())
