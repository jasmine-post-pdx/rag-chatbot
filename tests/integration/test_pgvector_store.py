"""Opt-in Postgres/pgvector store checks.

Run with a reachable pgvector instance, for example:

    docker compose --profile vector-db up -d postgres
    RUN_VECTOR_DB=1 .venv/bin/pytest -m vector_db
"""

import os

import pytest

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.document_store import DocumentChunk
from app.retrieval.pgvector_store import PgvectorDocumentStore
from app.retrieval.vector_store import EmbeddedChunk

pytestmark = [
    pytest.mark.vector_db,
    pytest.mark.skipif(
        os.getenv("RUN_VECTOR_DB") != "1",
        reason="Set RUN_VECTOR_DB=1 with a reachable pgvector instance",
    ),
]


def _dsn() -> str:
    """Build a DSN from environment defaults used by local Compose.

    :returns: Postgres connection URL."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "raglab")
    password = os.getenv("POSTGRES_PASSWORD", "raglab")
    database = os.getenv("POSTGRES_DB", "raglab")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def test_pgvector_upsert_search_and_delete_source():
    """Upsert, cosine search, and delete-by-source against a live pgvector DB."""
    embedder = FakeEmbeddingClient(dimensions=8, model="fake-embed")
    store = PgvectorDocumentStore(
        _dsn(),
        embedder,
        collection="test-step6",
        max_limit=5,
        dimensions=8,
    )
    store.delete_source("pg-demo")
    chunk = DocumentChunk(
        source_id="pg-demo",
        chunk_id="pg-demo:0",
        text="FastAPI lifespan handlers create shared clients on startup",
        title="PG Demo",
        framework="fastapi",
        language="python",
        metadata={"path": "data/raw/fastapi-lifespan.md"},
    )
    store.upsert_chunks(
        [EmbeddedChunk(chunk=chunk, embedding=embedder.embed_documents([chunk.text])[0])]
    )

    hits = store.search_documents("FastAPI lifespan clients", limit=3)
    assert hits
    assert hits[0].chunk_id == "pg-demo:0"
    assert hits[0].score is not None
    assert store.get_document_chunk("pg-demo:0") is not None

    removed = store.delete_source("pg-demo")
    assert removed >= 1
    assert store.get_document_chunk("pg-demo:0") is None
    assert store.is_ready() is True
