# Chunking strategy experiment (Step 7)

## Decision

Keep **`recursive_character`** as the production default (`CHUNKING_STRATEGY`).
On the current non-secret sample corpus and fake-embedder run, all three
strategies tied on every recorded metric, so there is no empirical winner to
promote yet. Prefer `structure_aware` only when re-indexing longer Markdown
with clear heading/code boundaries and after re-running the experiment with a
real embedding model.

## Experiment command (completed)

```bash
.venv/bin/python -m scripts.chunking_experiment \
  --path data/raw \
  --output docs/evals/chunking-experiment-results.json
```

Artifact: [`docs/evals/chunking-experiment-results.json`](../evals/chunking-experiment-results.json).

## Configuration actually used

| Field | Value |
| --- | --- |
| Corpus | `data/raw/*.md` (fastapi-guide, fastapi-lifespan, rag-budget, express-guide) |
| Queries | `tests/evals/retrieval_queries.json` (30 queries) |
| Embedding | `FakeEmbeddingClient` / `fake-embed` / 32 dims |
| Vector store | Fresh in-memory stores only (`vector_collection=chunking_experiment`) |
| top_k | 5 |
| Budget token limit | 500 |
| Recursive / structure size | `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50` |
| Token-aware size | `125` tokens max, overlap `12` (derived from settings ÷ 4) |

The live `documents` collection was not cleared or written.

## Results (from the JSON artifact)

| Strategy | Chunks | Answer-bearing top-k | Budget fit | Coherence | Citation clarity | Avg crowding |
| --- | --- | --- | --- | --- | --- | --- |
| recursive_character | 4 | 30/30 | 30/30 | 30/30 | 30/30 | 0.000 |
| token_aware | 4 | 30/30 | 30/30 | 30/30 | 30/30 | 0.000 |
| structure_aware | 4 | 30/30 | 30/30 | 30/30 | 30/30 | 0.000 |

### Example successes

- Every fixture query retrieved an expected `source_id` in top-k under all three
  strategies with the fake embedder.
- Context-budget fit stayed true for every query (small chunks, budget 500).
- Near-duplicate crowding stayed at 0.0 (top-k hits were not near-copies).

### Example failures / limits

- The sample Markdown files are short enough that each source collapses to a
  single chunk under size 500 for every strategy, so the bake-off could not
  show heading-boundary or token-cap differences.
- Fake-embedder ranking is a deterministic stand-in, not proof of semantic
  quality with `sentence-transformers` or a hosted embedding model.
- No strategy produced a miss to inspect; differentiation is deferred until a
  larger corpus or live embeddings is used.

## Tradeoffs

| Strategy | Strength | Cost |
| --- | --- | --- |
| recursive_character | Simple, stable IDs, matches Step 6 behavior | Can split mid-sentence / mid-section |
| token_aware | Honors model token budgets explicitly | Needs a counter; phrase boundaries still approximate |
| structure_aware | Keeps headings and fenced code together when they fit | Depends on Markdown structure; oversized sections still window |

## Chosen approach

**Default: `recursive_character`.** Metrics do not justify changing production
ingestion on this corpus. Revisit after expanding `data/raw` or running with
`EMBEDDING_PROVIDER=local`.

## Re-index instructions

- **Changed source:** `IngestionPipeline` deletes by `source_id`, then upserts
  new chunks for that source only.
- **Strategy change:** set `CHUNKING_STRATEGY`, then re-ingest
  (`python -m scripts.ingest` or restart lifespan seed). Do not leave mixed
  strategy IDs for the same source without a delete-before-write pass.
- **Experiment path:** `python -m scripts.chunking_experiment` always uses
  isolated in-memory indexes named `chunking_experiment` and refuses
  `--clear-collection documents`.
- **Full rebuild:** explicit only (clear/rebuild the chosen backend, then
  ingest). Never as a side effect of a single-source upsert.
