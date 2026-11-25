"""Unit tests for the in-memory vector document store."""

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.document_store import DocumentChunk, DocumentStore, SearchHit
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.retrieval.vector_store import EmbeddedChunk


def _store(*, max_limit: int = 5) -> InMemoryVectorDocumentStore:
    """Build an empty in-memory vector store with a fake embedder.

    :param max_limit: Hard maximum search results.
    :returns: A fresh store."""
    return InMemoryVectorDocumentStore(
        FakeEmbeddingClient(dimensions=16, model="fake-embed"),
        max_limit=max_limit,
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "src",
    title: str = "Title",
    framework: str | None = None,
    language: str | None = None,
    metadata: dict[str, str] | None = None,
) -> DocumentChunk:
    """Build a document chunk for upsert tests.

    :returns: A ``DocumentChunk`` instance."""
    return DocumentChunk(
        source_id=source_id,
        chunk_id=chunk_id,
        text=text,
        title=title,
        framework=framework,
        language=language,
        metadata=metadata or {"path": f"data/raw/{source_id}.md"},
    )


def test_empty_store_search_returns_empty_list():
    """Searching an empty store returns no hits, not an error."""
    store = _store()

    assert store.search_documents("FastAPI", limit=5) == []
    assert store.get_document_chunk("missing") is None


def test_search_returns_top_k_ordered_by_cosine_score():
    """Hits are ranked by descending cosine similarity and capped by limit."""
    store = _store(max_limit=10)
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    chunks = [
        _chunk("a:0", "FastAPI dependency injection with Depends", source_id="a"),
        _chunk("b:0", "Express middleware request ordering", source_id="b"),
        _chunk("c:0", "FastAPI lifespan shared clients", source_id="c"),
    ]
    embedded = [
        EmbeddedChunk(chunk=chunk, embedding=embedder.embed_documents([chunk.text])[0])
        for chunk in chunks
    ]
    store.upsert_chunks(embedded)

    hits = store.search_documents("dependency injection with Depends", limit=2)

    assert len(hits) == 2
    assert all(isinstance(hit, SearchHit) for hit in hits)
    assert all(hit.score is not None for hit in hits)
    assert hits[0].score >= hits[1].score
    assert hits[0].chunk_id == "a:0"
    assert hits[0].rank == 1
    assert hits[1].rank == 2


def test_search_supports_framework_and_language_filters():
    """Metadata filters restrict the candidate set before ranking."""
    store = _store()
    embedder = FakeEmbeddingClient(dimensions=16)
    chunks = [
        _chunk(
            "py:0",
            "FastAPI routers use dependency injection",
            source_id="py",
            framework="fastapi",
            language="python",
        ),
        _chunk(
            "ts:0",
            "Express routers use middleware injection patterns",
            source_id="ts",
            framework="express",
            language="typescript",
        ),
    ]
    store.upsert_chunks(
        [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedder.embed_documents([chunk.text])[0],
            )
            for chunk in chunks
        ]
    )

    hits = store.search_documents(
        "injection",
        framework="fastapi",
        language="python",
        limit=5,
    )

    assert [hit.chunk_id for hit in hits] == ["py:0"]


def test_search_hits_include_provenance_fields():
    """Hits expose chunk ID, score, source, title, metadata, and text."""
    store = _store()
    embedder = FakeEmbeddingClient(dimensions=16)
    chunk = _chunk(
        "fastapi-guide:0",
        "FastAPI lifespan handlers create shared clients",
        source_id="fastapi-guide",
        title="FastAPI Guide",
        framework="fastapi",
        language="python",
        metadata={"path": "data/raw/fastapi-guide.md", "version": "1"},
    )
    store.upsert_chunks(
        [EmbeddedChunk(chunk=chunk, embedding=embedder.embed_documents([chunk.text])[0])]
    )

    hits = store.search_documents("lifespan handlers", limit=1)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.chunk_id == "fastapi-guide:0"
    assert hit.source_id == "fastapi-guide"
    assert hit.title == "FastAPI Guide"
    assert hit.metadata == {"path": "data/raw/fastapi-guide.md", "version": "1"}
    assert "lifespan" in hit.text
    assert isinstance(hit.score, float)


def test_get_document_chunk_returns_chunk_or_none():
    """Direct lookup returns the stored chunk or None when missing."""
    store = _store()
    embedder = FakeEmbeddingClient(dimensions=16)
    chunk = _chunk("rag-notes:0", "Context budgeting reserves tokens", source_id="rag-notes")
    store.upsert_chunks(
        [EmbeddedChunk(chunk=chunk, embedding=embedder.embed_documents([chunk.text])[0])]
    )

    found = store.get_document_chunk("rag-notes:0")
    assert found is not None
    assert found.text.startswith("Context budgeting")
    assert store.get_document_chunk("missing") is None


def test_delete_source_removes_only_that_source():
    """Deleting a source removes its chunks and leaves others intact."""
    store = _store()
    embedder = FakeEmbeddingClient(dimensions=16)
    keep = _chunk("keep:0", "Express middleware attaches request IDs", source_id="keep")
    drop = _chunk("drop:0", "FastAPI Depends injects services", source_id="drop")
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=keep, embedding=embedder.embed_documents([keep.text])[0]),
            EmbeddedChunk(chunk=drop, embedding=embedder.embed_documents([drop.text])[0]),
        ]
    )

    removed = store.delete_source("drop")

    assert removed == 1
    assert store.get_document_chunk("drop:0") is None
    assert store.get_document_chunk("keep:0") is not None


def test_vector_store_satisfies_document_store_protocol():
    """The in-memory vector adapter is usable as a DocumentStore."""
    store: DocumentStore = _store()
    assert store.search_documents("x", limit=1) == []


def test_blank_query_returns_empty_hits():
    """Blank queries yield no hits without raising."""
    store = _store()
    embedder = FakeEmbeddingClient(dimensions=16)
    chunk = _chunk("a:0", "FastAPI dependency injection", source_id="a")
    store.upsert_chunks(
        [EmbeddedChunk(chunk=chunk, embedding=embedder.embed_documents([chunk.text])[0])]
    )

    assert store.search_documents("   ", limit=5) == []
