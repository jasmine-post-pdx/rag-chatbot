"""In-process tests for the retrieval-only search HTTP API."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.dependencies.search import get_search_service
from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import SourceDocument
from app.retrieval.ingestion import IngestionPipeline
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.services.search import SearchService


def _seeded_search_service() -> SearchService:
    """Build a search service with a small ingested corpus.

    :returns: A ``SearchService`` backed by an in-memory vector store."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    store = InMemoryVectorDocumentStore(embedder, max_limit=10)
    pipeline = IngestionPipeline(store, embedder, chunk_size=200, chunk_overlap=20)
    pipeline.ingest_documents(
        [
            SourceDocument(
                source_id="fastapi-guide",
                title="FastAPI Guide",
                text=(
                    "FastAPI dependency injection wires routers to services. "
                    "Use Depends to inject shared resources into path operations."
                ),
                framework="fastapi",
                language="python",
                metadata={"path": "data/raw/fastapi-guide.md"},
            ),
            SourceDocument(
                source_id="express-guide",
                title="Express Guide",
                text=(
                    "Express middleware runs in order for each request and can "
                    "attach request IDs or reject invalid input."
                ),
                framework="express",
                language="typescript",
                metadata={"path": "data/raw/express-guide.md"},
            ),
        ]
    )
    return SearchService(
        store,
        embedding_model=embedder.model,
        similarity_metric="cosine",
        default_top_k=5,
        max_top_k=10,
    )


@pytest.fixture
def search_client(client: TestClient):
    """Yield the shared test client with a seeded search service override.

    :param client: Base FastAPI test client fixture.
    :yields: The same client with search dependency overridden."""
    service = _seeded_search_service()
    main.app.dependency_overrides[get_search_service] = lambda: service
    try:
        yield client
    finally:
        main.app.dependency_overrides.pop(get_search_service, None)


@pytest.fixture
def empty_search_client(client: TestClient):
    """Yield a client whose search store is empty.

    :param client: Base FastAPI test client fixture.
    :yields: The client with an empty search service."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    store = InMemoryVectorDocumentStore(embedder, max_limit=10)
    service = SearchService(
        store,
        embedding_model=embedder.model,
        similarity_metric="cosine",
        default_top_k=5,
        max_top_k=10,
    )
    main.app.dependency_overrides[get_search_service] = lambda: service
    try:
        yield client
    finally:
        main.app.dependency_overrides.pop(get_search_service, None)


def test_search_returns_hits_with_provenance(search_client: TestClient):
    """POST /v1/search returns ranked hits with retrieval provenance fields."""
    response = search_client.post(
        "/v1/search",
        json={"query": "FastAPI Depends injection", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_model"] == "fake-embed"
    assert body["similarity_metric"] == "cosine"
    assert "answer" not in body
    assert "citations" not in body
    assert isinstance(body["hits"], list)
    assert body["hits"]
    hit = body["hits"][0]
    assert set(hit) >= {
        "chunk_id",
        "score",
        "source",
        "title",
        "metadata",
        "text",
    }
    assert hit["source"] == "fastapi-guide"
    assert isinstance(hit["score"], float)
    assert "Depends" in hit["text"] or "dependency" in hit["text"].lower()


def test_search_supports_metadata_filters(search_client: TestClient):
    """Optional framework and language filters narrow returned hits."""
    response = search_client.post(
        "/v1/search",
        json={
            "query": "middleware request",
            "framework": "express",
            "language": "typescript",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits
    assert all(hit["source"] == "express-guide" for hit in hits)


def test_search_empty_store_returns_empty_hits(empty_search_client: TestClient):
    """An empty index returns 200 with an empty hits list."""
    response = empty_search_client.post(
        "/v1/search",
        json={"query": "anything relevant"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hits"] == []
    assert body["similarity_metric"] == "cosine"


def test_search_rejects_missing_query(search_client: TestClient):
    """Missing query bodies are client validation errors."""
    response = search_client.post("/v1/search", json={})

    assert response.status_code == 422
    assert "detail" in response.json()


def test_search_rejects_blank_query(search_client: TestClient):
    """Blank queries are rejected with a clear 4xx response."""
    response = search_client.post("/v1/search", json={"query": "   "})

    assert response.status_code == 422


def test_search_rejects_top_k_above_configured_max(search_client: TestClient):
    """top_k above the service max is a client validation error."""
    response = search_client.post(
        "/v1/search",
        json={"query": "FastAPI", "top_k": 50},
    )

    assert response.status_code == 422


def test_search_respects_top_k(search_client: TestClient):
    """The response contains at most top_k hits."""
    response = search_client.post(
        "/v1/search",
        json={"query": "request", "top_k": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["hits"]) <= 1
