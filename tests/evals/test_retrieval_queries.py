"""Automated retrieval checks against the representative query fixture."""

import json
from pathlib import Path

import pytest

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import SourceDocument
from app.retrieval.ingestion import IngestionPipeline
from app.retrieval.memory_vector import InMemoryVectorDocumentStore

_FIXTURE = Path(__file__).with_name("retrieval_queries.json")


def _sample_sources() -> list[SourceDocument]:
    """Return the non-secret corpus mirrored by ``data/raw``.

    :returns: Source documents for deterministic eval ingestion."""
    return [
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
            source_id="fastapi-lifespan",
            title="FastAPI Lifespan",
            text=(
                "FastAPI lifespan handlers create shared clients on startup "
                "and close them on shutdown. Close OpenAI clients during FastAPI shutdown."
            ),
            framework="fastapi",
            language="python",
            metadata={"path": "data/raw/fastapi-lifespan.md"},
        ),
        SourceDocument(
            source_id="rag-budget",
            title="RAG Context Budget",
            text=(
                "Context budgeting reserves tokens for instructions, the question, "
                "tool schemas, output reserve, and a safety margin before selecting "
                "ranked chunks. Select ranked chunks without exceeding budget. "
                "Do not silently send oversized prompts. Context window allocation "
                "for retrieval happens before generation in a RAG service."
            ),
            framework=None,
            language="python",
            metadata={"path": "data/raw/rag-budget.md"},
        ),
        SourceDocument(
            source_id="express-guide",
            title="Express Guide",
            text=(
                "Express middleware runs in order for each request and can "
                "attach request IDs or reject invalid input. Reject malformed "
                "requests with Express middleware. TypeScript Express middleware pipeline."
            ),
            framework="express",
            language="typescript",
            metadata={"path": "data/raw/express-guide.md"},
        ),
    ]


@pytest.fixture(scope="module")
def retrieval_store() -> InMemoryVectorDocumentStore:
    """Ingest the sample corpus once for all retrieval query assertions.

    :returns: A populated in-memory vector store."""
    embedder = FakeEmbeddingClient(dimensions=32, model="fake-embed")
    store = InMemoryVectorDocumentStore(embedder, max_limit=10)
    pipeline = IngestionPipeline(store, embedder, chunk_size=400, chunk_overlap=40)
    pipeline.ingest_documents(_sample_sources())
    return store


def test_retrieval_fixture_has_twenty_to_thirty_queries():
    """The representative query set stays within the Step 6 size band."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    queries = payload["queries"]
    assert 20 <= len(queries) <= 30


def test_each_retrieval_query_returns_an_expected_source(
    retrieval_store: InMemoryVectorDocumentStore,
):
    """Every fixture query retrieves at least one expected source in top-k."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    top_k = int(payload["top_k"])
    failures: list[str] = []

    for item in payload["queries"]:
        hits = retrieval_store.search_documents(item["query"], limit=top_k)
        hit_sources = {hit.source_id for hit in hits}
        expected = set(item["expect_any_source_ids"])
        if hit_sources.isdisjoint(expected):
            failures.append(
                f"{item['id']}: query={item['query']!r} "
                f"expected_any={sorted(expected)} got={sorted(hit_sources)}"
            )

    assert failures == [], "\n".join(failures)
