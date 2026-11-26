"""Isolated multi-strategy chunking experiment harness (Step 7)."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.llm.embeddings import EmbeddingClient
from app.retrieval.chunking import ChunkingConfig, ChunkingStrategy, SourceDocument
from app.retrieval.document_store import SearchHit
from app.retrieval.ingestion import IngestionPipeline
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.services.token_budget import ApproximateTokenCounter

EXPERIMENT_COLLECTION = "chunking_experiment"
_NEAR_DUPLICATE_THRESHOLD = 0.9
_MARKDOWN_STARTERS = frozenset("#*`>-+|[(")


class QueryMetrics(BaseModel):
    """Per-query comparison metrics for one chunking strategy.

    :ivar answer_bearing_in_top_k: Whether an expected source appears in top-k.
    :ivar standalone_coherence: Whether the top hit looks boundary-aligned.
    :ivar citation_clarity: Whether every hit has usable chunk/source IDs.
    :ivar context_budget_fit: Whether top-k token sum fits the budget limit.
    :ivar near_duplicate_crowding: Fraction of near-duplicate hit pairs.
    :ivar hit_chunk_ids: Chunk IDs returned for the query."""

    answer_bearing_in_top_k: bool
    standalone_coherence: bool
    citation_clarity: bool
    context_budget_fit: bool
    near_duplicate_crowding: float
    hit_chunk_ids: list[str] = Field(default_factory=list)


class StrategyExperimentResult(BaseModel):
    """Aggregated experiment output for a single chunking strategy.

    :ivar settings: Chunking configuration used for this strategy.
    :ivar chunk_count: Number of chunks upserted into the isolated store.
    :ivar query_metrics: Metrics keyed by query id."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: ChunkingConfig
    chunk_count: int
    query_metrics: dict[str, QueryMetrics] = Field(default_factory=dict)


class ChunkingExperimentReport(BaseModel):
    """Full multi-strategy experiment report.

    :ivar embedding_model: Embedding model identity used for the run.
    :ivar top_k: Retrieval depth applied to every strategy.
    :ivar budget_token_limit: Token budget used for context-fit checks.
    :ivar vector_collection: Isolated experiment collection name.
    :ivar results_by_strategy: Per-strategy metrics and settings.
    :ivar stores_by_strategy: In-memory stores retained for isolation tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_model: str
    top_k: int
    budget_token_limit: int
    vector_collection: str = EXPERIMENT_COLLECTION
    results_by_strategy: dict[ChunkingStrategy, StrategyExperimentResult] = Field(
        default_factory=dict
    )
    stores_by_strategy: dict[ChunkingStrategy, Any] = Field(default_factory=dict)

    def safe_summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary without chunk body text.

        :returns: Nested dict of settings, IDs, counts, and metric flags."""
        results: dict[str, Any] = {}
        for strategy, result in self.results_by_strategy.items():
            query_metrics: dict[str, Any] = {}
            for query_id, metrics in result.query_metrics.items():
                query_metrics[query_id] = {
                    "answer_bearing_in_top_k": metrics.answer_bearing_in_top_k,
                    "standalone_coherence": metrics.standalone_coherence,
                    "citation_clarity": metrics.citation_clarity,
                    "context_budget_fit": metrics.context_budget_fit,
                    "near_duplicate_crowding": metrics.near_duplicate_crowding,
                    "hit_chunk_ids": list(metrics.hit_chunk_ids),
                }
            results[strategy.value] = {
                "settings": {
                    "strategy": result.settings.strategy.value,
                    "chunk_size": result.settings.chunk_size,
                    "chunk_overlap": result.settings.chunk_overlap,
                },
                "chunk_count": result.chunk_count,
                "query_metrics": query_metrics,
            }
        return {
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "budget_token_limit": self.budget_token_limit,
            "vector_collection": self.vector_collection,
            "collection": self.vector_collection,
            "results_by_strategy": results,
        }


def _normalize_for_similarity(text: str) -> str:
    """Collapse whitespace for near-duplicate comparisons.

    :param text: Chunk text.
    :returns: Normalized comparison string."""
    return " ".join(text.split()).casefold()


def _standalone_coherence(hits: Sequence[SearchHit]) -> bool:
    """Return whether the top hit appears boundary-aligned.

    :param hits: Ranked search hits.
    :returns: ``True`` when there are no hits or the top hit looks coherent."""
    if not hits:
        return True
    text = hits[0].text.lstrip()
    if not text:
        return True
    first = text[0]
    return first.isalnum() or first in _MARKDOWN_STARTERS


def _citation_clarity(hits: Sequence[SearchHit]) -> bool:
    """Return whether every hit has usable citation identifiers.

    :param hits: Ranked search hits.
    :returns: ``True`` when all hits expose non-empty chunk and source IDs."""
    if not hits:
        return True
    return all(bool(hit.chunk_id) and bool(hit.source_id) for hit in hits)


def _near_duplicate_crowding(hits: Sequence[SearchHit]) -> float:
    """Fraction of unordered hit pairs that are near-duplicates.

    :param hits: Ranked search hits.
    :returns: Crowding ratio in ``[0.0, 1.0]``."""
    if len(hits) < 2:
        return 0.0
    normalized = [_normalize_for_similarity(hit.text) for hit in hits]
    pair_count = 0
    duplicate_count = 0
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            pair_count += 1
            ratio = SequenceMatcher(None, normalized[left], normalized[right]).ratio()
            if ratio >= _NEAR_DUPLICATE_THRESHOLD:
                duplicate_count += 1
    return duplicate_count / pair_count if pair_count else 0.0


def _score_query(
    hits: Sequence[SearchHit],
    *,
    expect_any_source_ids: Sequence[str],
    budget_token_limit: int,
    token_counter: ApproximateTokenCounter,
) -> QueryMetrics:
    """Compute comparison metrics for one query's hits.

    :param hits: Ranked search hits.
    :param expect_any_source_ids: Acceptable source IDs for answer-bearing checks.
    :param budget_token_limit: Max tokens for context-budget fit.
    :param token_counter: Counter used for budget math.
    :returns: Populated ``QueryMetrics``."""
    expected = set(expect_any_source_ids)
    hit_sources = {hit.source_id for hit in hits}
    token_sum = sum(token_counter.count(hit.text) for hit in hits)
    return QueryMetrics(
        answer_bearing_in_top_k=bool(hit_sources & expected),
        standalone_coherence=_standalone_coherence(hits),
        citation_clarity=_citation_clarity(hits),
        context_budget_fit=token_sum <= budget_token_limit,
        near_duplicate_crowding=_near_duplicate_crowding(hits),
        hit_chunk_ids=[hit.chunk_id for hit in hits],
    )


def run_chunking_experiment(
    sources: list[SourceDocument],
    queries: Sequence[dict[str, Any]],
    *,
    strategies: Sequence[ChunkingConfig],
    embedder: EmbeddingClient,
    top_k: int,
    budget_token_limit: int,
    production_store: Any | None = None,
) -> ChunkingExperimentReport:
    """Run the same queries against isolated indexes for each strategy.

    Creates one fresh in-memory store per strategy. ``production_store`` is
    accepted for API compatibility and is never written.

    :param sources: Corpus documents to ingest per strategy.
    :param queries: Query dicts with ``id``, ``query``, ``expect_any_source_ids``.
    :param strategies: Chunking configs to compare.
    :param embedder: Embedding client shared for ingest and search.
    :param top_k: Retrieval depth for every query.
    :param budget_token_limit: Token budget for context-fit metrics.
    :param production_store: Ignored; must not be mutated.
    :returns: A populated experiment report."""
    del production_store  # Explicitly unused: isolation guarantee.

    token_counter = ApproximateTokenCounter()
    results_by_strategy: dict[ChunkingStrategy, StrategyExperimentResult] = {}
    stores_by_strategy: dict[ChunkingStrategy, InMemoryVectorDocumentStore] = {}

    for config in strategies:
        store = InMemoryVectorDocumentStore(embedder, max_limit=max(top_k, 1))
        # Mark isolation explicitly for collection-name checks.
        store.collection = EXPERIMENT_COLLECTION
        pipeline = IngestionPipeline(store, embedder, chunking_config=config)
        ingest_result = pipeline.ingest_documents(list(sources))

        query_metrics: dict[str, QueryMetrics] = {}
        for item in queries:
            hits = store.search_documents(item["query"], limit=top_k)
            query_metrics[str(item["id"])] = _score_query(
                hits,
                expect_any_source_ids=item.get("expect_any_source_ids", []),
                budget_token_limit=budget_token_limit,
                token_counter=token_counter,
            )

        results_by_strategy[config.strategy] = StrategyExperimentResult(
            settings=config,
            chunk_count=ingest_result.chunks_upserted,
            query_metrics=query_metrics,
        )
        stores_by_strategy[config.strategy] = store

    return ChunkingExperimentReport(
        embedding_model=embedder.model,
        top_k=top_k,
        budget_token_limit=budget_token_limit,
        vector_collection=EXPERIMENT_COLLECTION,
        results_by_strategy=results_by_strategy,
        stores_by_strategy=stores_by_strategy,
    )
