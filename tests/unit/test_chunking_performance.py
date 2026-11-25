"""Performance bounds for Step 7 chunking strategies and experiment runs."""

from __future__ import annotations

import math
import time

from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    SourceDocument,
    chunk_document,
)
from app.retrieval.chunking_experiment import run_chunking_experiment
from app.services.token_budget import ApproximateTokenCounter


def _large_source(*, chars: int = 100_000) -> SourceDocument:
    """Build a large synthetic document for wall-clock chunking bounds.

    :param chars: Approximate body size in characters.
    :returns: A ``SourceDocument`` near the requested size."""
    unit = "alpha beta gamma delta epsilon zeta eta theta "
    repeats = max(1, chars // len(unit))
    text = (unit * repeats)[:chars]
    return SourceDocument(
        source_id="large-doc",
        title="Large",
        text=text,
        metadata={"path": "data/raw/large-doc.md", "version": "1"},
    )


def _strategy_configs() -> list[ChunkingConfig]:
    """Return configs for all three strategies used in performance checks.

    :returns: Recursive, token-aware, and structure-aware configs."""
    return [
        ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=500,
            chunk_overlap=50,
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN_AWARE,
            chunk_size=120,
            chunk_overlap=12,
            token_counter=ApproximateTokenCounter(),
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            chunk_size=500,
            chunk_overlap=50,
        ),
    ]


def test_large_document_chunking_stays_within_time_and_count_bounds():
    """Each strategy chunks ~100k chars quickly with a bounded chunk count."""
    source = _large_source(chars=100_000)
    for config in _strategy_configs():
        started = time.perf_counter()
        chunks = chunk_document(source, config=config)
        elapsed = time.perf_counter() - started

        assert elapsed < 2.0, f"{config.strategy} took {elapsed:.3f}s"
        assert chunks

        if config.strategy == ChunkingStrategy.TOKEN_AWARE:
            # ApproximateTokenCounter ≈ ceil(len/4); size is max tokens.
            approx_tokens = (len(source.text) + 3) // 4
            step = max(1, config.chunk_size - config.chunk_overlap)
            upper = math.ceil(approx_tokens / step) + 5
        else:
            step = max(1, config.chunk_size - config.chunk_overlap)
            upper = math.ceil(len(source.text) / step) + 5
        assert len(chunks) <= upper


def test_full_experiment_over_sample_queries_stays_within_time_bound():
    """Corpus × ~30 queries × 3 strategies with fake embeddings finishes quickly."""
    sources = [
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
                "and close them on shutdown."
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
                "tool schemas, output reserve, and a safety margin."
            ),
            language="python",
            metadata={"path": "data/raw/rag-budget.md"},
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
    queries = [
        {
            "id": f"q{index:02d}",
            "query": f"representative retrieval question number {index}",
            "expect_any_source_ids": ["fastapi-guide", "rag-budget", "express-guide"],
        }
        for index in range(1, 31)
    ]

    started = time.perf_counter()
    report = run_chunking_experiment(
        sources,
        queries,
        strategies=_strategy_configs(),
        embedder=FakeEmbeddingClient(dimensions=16, model="fake-embed"),
        top_k=5,
        budget_token_limit=400,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"experiment took {elapsed:.3f}s"
    assert len(report.results_by_strategy) == 3
    # Sequential isolated stores: three strategy entries, no shared mutation surface.
    assert len(report.stores_by_strategy) == 3
    assert len({id(store) for store in report.stores_by_strategy.values()}) == 3


def test_near_duplicate_crowding_on_small_top_k_stays_cheap():
    """Crowding over top-k=5 remains inside the overall experiment time budget."""
    sources = [
        SourceDocument(
            source_id="dup-heavy",
            title="Duplicates",
            text=(
                "Context budgeting reserves tokens for instructions. "
                "Context budgeting reserves tokens for instructions. "
                "Context budgeting reserves tokens for instructions. "
                "Context budgeting reserves tokens for instructions."
            ),
            metadata={"path": "data/raw/dup-heavy.md"},
        )
    ]
    queries = [
        {
            "id": "q-dup",
            "query": "Context budgeting reserves tokens",
            "expect_any_source_ids": ["dup-heavy"],
        }
    ]

    started = time.perf_counter()
    report = run_chunking_experiment(
        sources,
        queries,
        strategies=_strategy_configs(),
        embedder=FakeEmbeddingClient(dimensions=8, model="fake-embed"),
        top_k=5,
        budget_token_limit=200,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    for result in report.results_by_strategy.values():
        crowding = result.query_metrics["q-dup"].near_duplicate_crowding
        assert 0.0 <= crowding <= 1.0
