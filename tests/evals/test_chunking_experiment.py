"""Eval harness: run all three chunking strategies on the representative queries.

Does not declare a winning strategy — only that metrics are fully populated so
``docs/decisions/chunking.md`` can later be written from real results.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import ChunkingConfig, ChunkingStrategy, SourceDocument
from app.retrieval.chunking_experiment import run_chunking_experiment
from app.services.token_budget import ApproximateTokenCounter

_FIXTURE = Path(__file__).with_name("retrieval_queries.json")
_PRODUCTION_COLLECTION = "documents"


def _sample_sources() -> list[SourceDocument]:
    """Return the non-secret corpus mirrored by ``data/raw``.

    :returns: Source documents for deterministic experiment ingestion."""
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


def _strategy_configs() -> list[ChunkingConfig]:
    """Return the three Step 7 variants under identical retrieval settings.

    :returns: Recursive, token-aware, and structure-aware configs."""
    return [
        ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=400,
            chunk_overlap=40,
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN_AWARE,
            chunk_size=100,
            chunk_overlap=10,
            token_counter=ApproximateTokenCounter(),
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            chunk_size=400,
            chunk_overlap=40,
        ),
    ]


def test_retrieval_fixture_still_has_twenty_to_thirty_queries():
    """The shared representative query set stays within the Step 6/7 size band."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert 20 <= len(payload["queries"]) <= 30


def test_chunking_experiment_populates_metrics_for_all_strategies_and_queries():
    """Every strategy×query cell has metrics; no winner is asserted."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    queries = payload["queries"]
    top_k = int(payload["top_k"])

    report = run_chunking_experiment(
        _sample_sources(),
        queries,
        strategies=_strategy_configs(),
        embedder=FakeEmbeddingClient(dimensions=32, model="fake-embed"),
        top_k=top_k,
        budget_token_limit=500,
    )

    expected_strategies = {
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.TOKEN_AWARE,
        ChunkingStrategy.STRUCTURE_AWARE,
    }
    assert set(report.results_by_strategy) == expected_strategies
    assert report.embedding_model == "fake-embed"
    assert report.top_k == top_k

    query_ids = {item["id"] for item in queries}
    for strategy, result in report.results_by_strategy.items():
        assert set(result.query_metrics) == query_ids, strategy
        assert result.chunk_count >= 1
        for metrics in result.query_metrics.values():
            assert isinstance(metrics.answer_bearing_in_top_k, bool)
            assert isinstance(metrics.standalone_coherence, bool)
            assert isinstance(metrics.citation_clarity, bool)
            assert isinstance(metrics.context_budget_fit, bool)
            assert 0.0 <= metrics.near_duplicate_crowding <= 1.0


def test_chunking_experiment_stores_are_isolated_from_production_collection_name():
    """Experiment indexes must not reuse the live ``documents`` collection identity."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    report = run_chunking_experiment(
        _sample_sources(),
        payload["queries"][:3],
        strategies=_strategy_configs(),
        embedder=FakeEmbeddingClient(dimensions=16, model="fake-embed"),
        top_k=3,
        budget_token_limit=300,
    )

    collection_names = set()
    if getattr(report, "vector_collection", None) is not None:
        collection_names.add(report.vector_collection)
    for store in report.stores_by_strategy.values():
        name = getattr(store, "collection", None) or getattr(
            store, "collection_name", None
        )
        if name is not None:
            collection_names.add(name)

    # Experiment indexes must never claim the live production collection name.
    assert _PRODUCTION_COLLECTION not in collection_names
    if hasattr(report, "vector_collection"):
        name = report.vector_collection
        assert name != _PRODUCTION_COLLECTION
        assert "experiment" in str(name).casefold() or str(name).startswith("chunking_")
