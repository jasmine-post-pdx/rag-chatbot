"""Unit tests for the read-only in-memory document store."""

from app.retrieval.document_store import DocumentChunk, DocumentStore
from app.retrieval.memory import InMemoryDocumentStore, sample_corpus


def _store(chunks: list[DocumentChunk] | None = None) -> InMemoryDocumentStore:
    """Build an in-memory store with the given chunks.

    :param chunks: Chunks to index; ``None`` uses the sample corpus.
    :returns: A populated ``InMemoryDocumentStore``."""
    if chunks is None:
        return InMemoryDocumentStore(sample_corpus())
    return InMemoryDocumentStore(chunks)


def test_search_by_query_returns_matching_chunks():
    """Return chunks whose text contains the query (case-insensitive)."""
    store = _store(
        [
            DocumentChunk(
                source_id="fastapi-guide",
                chunk_id="c1",
                text="FastAPI dependency injection wires routers to services.",
                framework="fastapi",
                language="python",
            ),
            DocumentChunk(
                source_id="django-guide",
                chunk_id="c2",
                text="Django models map to database tables.",
                framework="django",
                language="python",
            ),
        ]
    )

    hits = store.search_documents("dependency injection", limit=5)

    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].source_id == "fastapi-guide"
    assert "dependency injection" in hits[0].text.lower()


def test_search_filters_by_framework_and_language():
    """Optional framework and language filters narrow search results."""
    store = _store(
        [
            DocumentChunk(
                source_id="a",
                chunk_id="py-fastapi",
                text="async routes in fastapi",
                framework="fastapi",
                language="python",
            ),
            DocumentChunk(
                source_id="b",
                chunk_id="ts-express",
                text="async routes in express",
                framework="express",
                language="typescript",
            ),
            DocumentChunk(
                source_id="c",
                chunk_id="py-flask",
                text="async routes in flask",
                framework="flask",
                language="python",
            ),
        ]
    )

    hits = store.search_documents(
        "async routes",
        framework="fastapi",
        language="python",
        limit=5,
    )

    assert [hit.chunk_id for hit in hits] == ["py-fastapi"]


def test_search_respects_limit_and_hard_max():
    """Caller limit is honored and cannot exceed the store hard max."""
    chunks = [
        DocumentChunk(
            source_id="src",
            chunk_id=f"c{i}",
            text=f"shared token match {i}",
        )
        for i in range(10)
    ]
    store = InMemoryDocumentStore(chunks, max_limit=5)

    limited = store.search_documents("shared token", limit=3)
    assert len(limited) == 3

    capped = store.search_documents("shared token", limit=20)
    assert len(capped) == 5


def test_get_document_chunk_returns_chunk_or_none():
    """Lookup by chunk_id returns the chunk, or None when missing."""
    store = _store(
        [
            DocumentChunk(
                source_id="src",
                chunk_id="chunk-42",
                text="exact chunk body",
            )
        ]
    )

    found = store.get_document_chunk("chunk-42")
    assert found is not None
    assert found.chunk_id == "chunk-42"
    assert found.text == "exact chunk body"
    assert found.source_id == "src"

    assert store.get_document_chunk("missing") is None


def test_empty_store_and_no_matches_return_empty_list():
    """Empty store or unmatched query returns an empty list, not an error."""
    empty = InMemoryDocumentStore([])
    assert empty.search_documents("anything") == []

    store = _store(
        [
            DocumentChunk(
                source_id="src",
                chunk_id="c1",
                text="only about widgets",
            )
        ]
    )
    assert store.search_documents("gadgets") == []


def test_search_hits_are_compatible_with_ranked_chunk_fields():
    """Hits expose source_id, chunk_id, and text for RankedChunk mapping."""
    store = _store()
    hits = store.search_documents("fastapi", limit=1)
    assert hits, "sample corpus should match fastapi"
    hit = hits[0]
    assert hit.source_id
    assert hit.chunk_id
    assert hit.text
    # Structural compatibility with RankedChunk field names.
    ranked_fields = {"source_id", "chunk_id", "text"}
    assert ranked_fields.issubset(set(hit.model_dump()))


def test_inmemory_store_satisfies_document_store_protocol():
    """InMemoryDocumentStore structurally satisfies DocumentStore."""
    store: DocumentStore = InMemoryDocumentStore([])
    assert store.search_documents("q") == []
    assert store.get_document_chunk("x") is None


def test_sample_corpus_is_non_empty_and_deterministic():
    """Sample corpus provides a stable non-secret seed for tests."""
    first = sample_corpus()
    second = sample_corpus()
    assert len(first) >= 3
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
