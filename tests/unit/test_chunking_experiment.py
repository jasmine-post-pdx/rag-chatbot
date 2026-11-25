"""Unit tests for the isolated Step 7 chunking experiment harness."""

from __future__ import annotations

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import ChunkingConfig, ChunkingStrategy, SourceDocument
from app.retrieval.chunking_experiment import (
    ChunkingExperimentReport,
    run_chunking_experiment,
)
from app.retrieval.ingestion import IngestionPipeline
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.services.token_budget import ApproximateTokenCounter


def _sources() -> list[SourceDocument]:
    """Return a tiny non-secret corpus for experiment isolation tests.

    :returns: Two source documents with distinct content."""
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
            metadata={"path": "data/raw/fastapi-guide.md", "version": "1"},
        ),
        SourceDocument(
            source_id="rag-budget",
            title="RAG Context Budget",
            text=(
                "Context budgeting reserves tokens for instructions, the question, "
                "and a safety margin before selecting ranked chunks."
            ),
            language="python",
            metadata={"path": "data/raw/rag-budget.md", "version": "1"},
        ),
    ]


def _queries() -> list[dict]:
    """Return representative experiment queries with expected sources.

    :returns: Query dicts matching the eval fixture shape."""
    return [
        {
            "id": "q01",
            "query": "How does FastAPI dependency injection work?",
            "expect_any_source_ids": ["fastapi-guide"],
        },
        {
            "id": "q02",
            "query": "What is context budgeting in RAG?",
            "expect_any_source_ids": ["rag-budget"],
        },
    ]


def _configs() -> list[ChunkingConfig]:
    """Return the three Step 7 strategy configs under test.

    :returns: Recursive, token-aware, and structure-aware configs."""
    return [
        ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=120,
            chunk_overlap=20,
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN_AWARE,
            chunk_size=30,
            chunk_overlap=4,
            token_counter=ApproximateTokenCounter(),
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            chunk_size=120,
            chunk_overlap=20,
        ),
    ]


def test_experiment_uses_distinct_stores_per_strategy():
    """Each strategy gets an isolated index; A cannot see B's exclusive chunks."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    report = run_chunking_experiment(
        _sources(),
        _queries(),
        strategies=_configs(),
        embedder=embedder,
        top_k=3,
        budget_token_limit=200,
    )

    assert isinstance(report, ChunkingExperimentReport)
    stores = report.stores_by_strategy
    assert ChunkingStrategy.RECURSIVE_CHARACTER in stores
    assert ChunkingStrategy.TOKEN_AWARE in stores
    assert stores[ChunkingStrategy.RECURSIVE_CHARACTER] is not stores[
        ChunkingStrategy.TOKEN_AWARE
    ]

    recursive_ids = {
        hit.chunk_id
        for hit in stores[ChunkingStrategy.RECURSIVE_CHARACTER].search_documents(
            "dependency injection", limit=5
        )
    }
    # Mutate only the token-aware store and confirm recursive store is unchanged.
    token_store = stores[ChunkingStrategy.TOKEN_AWARE]
    pipeline = IngestionPipeline(
        token_store,
        embedder,
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN_AWARE,
            chunk_size=20,
            chunk_overlap=2,
            token_counter=ApproximateTokenCounter(),
        ),
    )
    pipeline.ingest_documents(
        [
            SourceDocument(
                source_id="exclusive-token-only",
                title="Exclusive",
                text="unique token-aware marker phrase zebra-quark-99",
            )
        ]
    )

    after_ids = {
        hit.chunk_id
        for hit in stores[ChunkingStrategy.RECURSIVE_CHARACTER].search_documents(
            "dependency injection", limit=5
        )
    }
    assert after_ids == recursive_ids
    exclusive = token_store.search_documents("zebra-quark-99", limit=5)
    assert exclusive
    assert all(hit.source_id == "exclusive-token-only" for hit in exclusive)


def test_experiment_applies_same_queries_and_settings_to_every_strategy():
    """Every strategy is evaluated with the same queries, top_k, and embedder."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed-exp")
    queries = _queries()
    report = run_chunking_experiment(
        _sources(),
        queries,
        strategies=_configs(),
        embedder=embedder,
        top_k=4,
        budget_token_limit=180,
    )

    assert report.embedding_model == "fake-embed-exp"
    assert report.top_k == 4
    assert report.budget_token_limit == 180
    assert len(report.results_by_strategy) == 3

    query_ids = {item["id"] for item in queries}
    for strategy, result in report.results_by_strategy.items():
        assert strategy in {
            ChunkingStrategy.RECURSIVE_CHARACTER,
            ChunkingStrategy.TOKEN_AWARE,
            ChunkingStrategy.STRUCTURE_AWARE,
        }
        assert result.chunk_count >= 1
        assert set(result.query_metrics) == query_ids
        assert result.settings.strategy == strategy
        assert result.settings.chunk_size >= 1


def test_experiment_report_metrics_are_populated_for_every_query():
    """Each query×strategy metric bundle exposes the Step 7 comparison fields."""
    report = run_chunking_experiment(
        _sources(),
        _queries(),
        strategies=_configs(),
        embedder=FakeEmbeddingClient(dimensions=16, model="fake-embed"),
        top_k=3,
        budget_token_limit=150,
    )

    for result in report.results_by_strategy.values():
        for metrics in result.query_metrics.values():
            assert isinstance(metrics.answer_bearing_in_top_k, bool)
            assert isinstance(metrics.standalone_coherence, bool)
            assert isinstance(metrics.citation_clarity, bool)
            assert isinstance(metrics.context_budget_fit, bool)
            assert isinstance(metrics.near_duplicate_crowding, float)
            assert 0.0 <= metrics.near_duplicate_crowding <= 1.0
            assert metrics.hit_chunk_ids is not None
            assert all(isinstance(chunk_id, str) for chunk_id in metrics.hit_chunk_ids)


def test_experiment_reindex_removes_stale_chunks_only_in_that_store():
    """Re-ingesting a source inside an experiment store clears stale IDs there only."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    report = run_chunking_experiment(
        _sources(),
        _queries(),
        strategies=[
            ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
                chunk_size=80,
                chunk_overlap=10,
            )
        ],
        embedder=embedder,
        top_k=5,
        budget_token_limit=200,
    )
    store = report.stores_by_strategy[ChunkingStrategy.RECURSIVE_CHARACTER]
    original_ids = {
        hit.chunk_id for hit in store.search_documents("budgeting", limit=10)
    }
    assert original_ids

    sibling = run_chunking_experiment(
        _sources(),
        _queries(),
        strategies=[
            ChunkingConfig(
                strategy=ChunkingStrategy.STRUCTURE_AWARE,
                chunk_size=80,
                chunk_overlap=10,
            )
        ],
        embedder=embedder,
        top_k=5,
        budget_token_limit=200,
    )
    sibling_store = sibling.stores_by_strategy[ChunkingStrategy.STRUCTURE_AWARE]
    sibling_before = {
        hit.chunk_id for hit in sibling_store.search_documents("budgeting", limit=10)
    }

    pipeline = IngestionPipeline(
        store,
        embedder,
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=80,
            chunk_overlap=10,
        ),
    )
    pipeline.ingest_documents(
        [
            SourceDocument(
                source_id="rag-budget",
                title="RAG Context Budget",
                text="Updated budgeting policy skips oversized chunks without silent truncation.",
                language="python",
                metadata={"path": "data/raw/rag-budget.md", "version": "2"},
            )
        ]
    )

    remaining = store.search_documents("budgeting", limit=10)
    remaining_ids = {hit.chunk_id for hit in remaining}
    assert "silent truncation" in " ".join(hit.text for hit in remaining)
    assert not any("safety margin" in hit.text for hit in remaining)
    for chunk_id in original_ids - remaining_ids:
        assert store.get_document_chunk(chunk_id) is None

    sibling_after = {
        hit.chunk_id for hit in sibling_store.search_documents("budgeting", limit=10)
    }
    assert sibling_after == sibling_before


def test_experiment_leaves_caller_supplied_production_store_unmodified():
    """A pre-existing production store must be unchanged after the experiment runs."""
    embedder = FakeEmbeddingClient(dimensions=16, model="fake-embed")
    production = InMemoryVectorDocumentStore(embedder, max_limit=10)
    pipeline = IngestionPipeline(production, embedder, chunk_size=100, chunk_overlap=10)
    pipeline.ingest_documents(_sources())
    before_ids = {
        hit.chunk_id for hit in production.search_documents("FastAPI", limit=10)
    }
    assert before_ids

    run_chunking_experiment(
        _sources(),
        _queries(),
        strategies=_configs(),
        embedder=embedder,
        top_k=3,
        budget_token_limit=200,
        production_store=production,
    )

    after_ids = {
        hit.chunk_id for hit in production.search_documents("FastAPI", limit=10)
    }
    assert after_ids == before_ids
    for chunk_id in before_ids:
        assert production.get_document_chunk(chunk_id) is not None
