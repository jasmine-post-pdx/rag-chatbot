"""CLI to compare chunking strategies on an isolated in-memory index."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config.settings import settings
from app.llm.embeddings import FakeEmbeddingClient
from app.retrieval.chunking import ChunkingConfig, ChunkingStrategy
from app.retrieval.chunking_experiment import (
    EXPERIMENT_COLLECTION,
    run_chunking_experiment,
)
from app.retrieval.ingestion import load_markdown_source
from app.services.token_budget import ApproximateTokenCounter

logger = logging.getLogger(__name__)

_DEFAULT_QUERIES = (
    Path(__file__).resolve().parents[1] / "tests" / "evals" / "retrieval_queries.json"
)
_PRODUCTION_COLLECTION = "documents"


def _strategy_configs(
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkingConfig]:
    """Build the three Step 7 strategy configs for an experiment run.

    :param chunk_size: Character/token size passed to each strategy.
    :param chunk_overlap: Overlap passed to each strategy.
    :returns: Recursive, token-aware, and structure-aware configs."""
    return [
        ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN_AWARE,
            chunk_size=max(1, chunk_size // 4),
            chunk_overlap=max(0, min(chunk_overlap // 4, (chunk_size // 4) - 1)),
            token_counter=ApproximateTokenCounter(),
        ),
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    ]


def _load_queries(path: Path) -> tuple[list[dict], int]:
    """Load representative retrieval queries from the eval fixture.

    :param path: JSON fixture path.
    :returns: Query list and default top_k."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["queries"]), int(payload.get("top_k", 5))


def main(argv: list[str] | None = None) -> int:
    """Run the chunking experiment and write a safe JSON report.

    :param argv: Optional CLI argument list.
    :returns: Process exit code.
    :raises SystemExit: When a destructive production clear is requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(settings.corpus_raw_dir),
        help="Directory of markdown sources (default: CORPUS_RAW_DIR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the safe JSON experiment report",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval depth (default: fixture top_k)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=_DEFAULT_QUERIES,
        help="Retrieval query fixture JSON",
    )
    parser.add_argument(
        "--clear-collection",
        type=str,
        default=None,
        help="Rejected when set to the live documents collection",
    )
    parser.add_argument(
        "--budget-token-limit",
        type=int,
        default=500,
        help="Token budget used for context-fit metrics",
    )
    args = parser.parse_args(argv)

    if args.clear_collection is not None:
        requested = args.clear_collection.strip().casefold()
        if requested == _PRODUCTION_COLLECTION:
            raise SystemExit(
                "Refusing to clear the live documents collection during a "
                "chunking experiment; use an isolated experiment path instead."
            )

    logging.basicConfig(level=settings.log_level)

    if not args.path.is_dir():
        raise SystemExit(f"Corpus path is not a directory: {args.path}")

    sources = [
        load_markdown_source(file_path)
        for file_path in sorted(args.path.glob("*.md"))
        if file_path.is_file()
    ]
    if not sources:
        raise SystemExit(f"No markdown sources found under {args.path}")

    queries, fixture_top_k = _load_queries(args.queries)
    top_k = args.top_k if args.top_k is not None else fixture_top_k

    embedder = FakeEmbeddingClient(
        dimensions=settings.embedding_dimensions,
        model=settings.embedding_model,
    )
    report = run_chunking_experiment(
        sources,
        queries,
        strategies=_strategy_configs(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedder=embedder,
        top_k=top_k,
        budget_token_limit=args.budget_token_limit,
    )

    summary = report.safe_summary()
    # Ensure collection identity is explicit for CLI contract tests.
    summary["vector_collection"] = EXPERIMENT_COLLECTION
    summary["collection"] = EXPERIMENT_COLLECTION

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Chunking experiment complete strategies=%s queries=%s output=%s collection=%s",
        len(report.results_by_strategy),
        len(queries),
        args.output,
        EXPERIMENT_COLLECTION,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
