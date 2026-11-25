"""Unit tests for the document ingestion pipeline."""

from pathlib import Path

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import SourceDocument
from app.retrieval.ingestion import IngestionPipeline, IngestResult
from app.retrieval.memory_vector import InMemoryVectorDocumentStore


def _pipeline(
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> tuple[IngestionPipeline, InMemoryVectorDocumentStore]:
    """Build an ingestion pipeline backed by an in-memory vector store.

    :returns: The pipeline and its store."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    store = InMemoryVectorDocumentStore(embedder, max_limit=10)
    pipeline = IngestionPipeline(
        store,
        embedder,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return pipeline, store


def test_ingest_documents_upserts_chunks_into_store():
    """Loading sources chunks, embeds, and makes them searchable."""
    pipeline, store = _pipeline()
    sources = [
        SourceDocument(
            source_id="fastapi-guide",
            title="FastAPI Guide",
            text="FastAPI dependency injection wires routers to services with Depends.",
            framework="fastapi",
            language="python",
        )
    ]

    result = pipeline.ingest_documents(sources)

    assert isinstance(result, IngestResult)
    assert result.sources_processed == 1
    assert result.chunks_upserted >= 1
    hits = store.search_documents("dependency injection", limit=5)
    assert hits
    assert hits[0].source_id == "fastapi-guide"
    assert store.get_document_chunk(hits[0].chunk_id) is not None


def test_reingest_same_source_replaces_stale_chunks():
    """Re-ingesting a source deletes prior chunks before upserting new ones."""
    pipeline, store = _pipeline(chunk_size=80, chunk_overlap=10)
    original = SourceDocument(
        source_id="rag-notes",
        title="RAG Notes",
        text="Context budgeting reserves tokens for instructions and output reserve.",
        language="python",
    )
    updated = SourceDocument(
        source_id="rag-notes",
        title="RAG Notes",
        text="Updated budgeting policy skips oversized chunks without silent truncation.",
        language="python",
    )

    pipeline.ingest_documents([original])
    original_ids = {
        hit.chunk_id for hit in store.search_documents("budgeting", limit=10)
    }
    assert original_ids

    second = pipeline.ingest_documents([updated])
    assert second.sources_deleted_before_upsert >= 1
    assert second.chunks_upserted >= 1

    remaining = store.search_documents("budgeting", limit=10)
    remaining_ids = {hit.chunk_id for hit in remaining}
    assert "silent truncation" in " ".join(hit.text for hit in remaining)
    assert not any("output reserve" in hit.text for hit in remaining)
    # Prior chunk IDs for this source must not remain searchable.
    stale = original_ids - remaining_ids
    for chunk_id in stale:
        assert store.get_document_chunk(chunk_id) is None


def test_delete_source_removes_all_chunks_for_source():
    """Explicit source deletion clears provenance without touching other sources."""
    pipeline, store = _pipeline()
    pipeline.ingest_documents(
        [
            SourceDocument(
                source_id="keep",
                title="Keep",
                text="Express middleware runs in order for each request.",
                framework="express",
                language="typescript",
            ),
            SourceDocument(
                source_id="drop",
                title="Drop",
                text="FastAPI lifespan creates shared clients on startup.",
                framework="fastapi",
                language="python",
            ),
        ]
    )

    removed = pipeline.delete_source("drop")

    assert removed >= 1
    assert store.get_document_chunk("drop:0") is None
    hits = store.search_documents("middleware", framework="express", limit=5)
    assert hits
    assert all(hit.source_id == "keep" for hit in hits)


def test_ingest_empty_corpus_is_a_no_op():
    """An empty source list upserts nothing and leaves the store empty."""
    pipeline, store = _pipeline()

    result = pipeline.ingest_documents([])

    assert result.sources_processed == 0
    assert result.chunks_upserted == 0
    assert store.search_documents("anything", limit=5) == []


def test_ingest_directory_loads_markdown_files(tmp_path: Path):
    """Directory ingestion loads ``.md`` files using the stem as source_id."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fastapi-guide.md").write_text(
        "---\ntitle: FastAPI Guide\nframework: fastapi\nlanguage: python\n---\n"
        "FastAPI Depends injects shared resources into path operations.\n",
        encoding="utf-8",
    )
    (raw / "notes.txt").write_text("ignored extension", encoding="utf-8")

    pipeline, store = _pipeline()
    result = pipeline.ingest_directory(raw)

    assert result.sources_processed == 1
    assert result.chunks_upserted >= 1
    hits = store.search_documents("Depends injects", limit=5)
    assert hits
    assert hits[0].source_id == "fastapi-guide"
    assert hits[0].title == "FastAPI Guide"
