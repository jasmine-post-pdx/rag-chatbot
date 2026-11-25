# Embeddings and retrieval (Step 6)

## Decision

Ship a retrieval-only path that loads markdown from `data/raw/`, normalizes and
chunks text, embeds with an application-owned `EmbeddingClient`, and upserts into
a vector-backed `DocumentStore`. Expose `POST /v1/search` with provenance fields
and no generation. Connect Step 5 document tools to the same store.

## Defaults

| Setting | Default | Notes |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `fake` | Deterministic hash embedder for local/CI without torch |
| `EMBEDDING_MODEL` | `fake-embed` | Reported in search responses |
| `EMBEDDING_DIMENSIONS` | `32` | Used by the fake provider |
| `VECTOR_STORE` | `memory` | In-process cosine index |
| `SIMILARITY_METRIC` | `cosine` | Score is similarity, not calibrated confidence |
| `VECTOR_COLLECTION` | `documents` | Logical collection name (pgvector rows) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Recursive character windows |

Optional local embeddings: set `EMBEDDING_PROVIDER=local`,
`EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, and install
`pip install '.[embeddings]'`. Optional durable store: set
`VECTOR_STORE=pgvector` and start Compose profile `vector-db`.

## Re-index policy

- **Changed source:** ingest deletes all chunks for that `source_id`, then
  upserts the new chunks. Stale chunk IDs for that source are not left searchable.
- **Deleted source:** call `IngestionPipeline.delete_source(source_id)` (or the
  store’s `delete_source`). No silent tombstones.
- **Full rebuild:** only via an explicit clear/rebuild of the chosen backend
  (drop/recreate the pgvector table or restart a memory-backed process and
  re-run `scripts/ingest.py`). The API never deletes an entire index as a side
  effect of a single source upsert.
- **Source originals:** files under `data/raw/` are preserved; ingestion reads
  them and does not delete them.

Stable chunk IDs are `{source_id}:{ordinal}` from the Step 6 character splitter.

## Score semantics

Returned `score` values are cosine similarity between the query embedding and
chunk embedding. Higher is more similar. They are **not** calibrated
probabilities or confidence.

## Tool wiring

When `GENERATION_PROVIDER=openai`, lifespan injects the same vector store into
the question processor that backs `POST /v1/search`. Allow-listed tools
`search_documents` and `get_document_chunk` therefore hit embedding retrieval
instead of the Step 5 substring stand-in. The substring
`InMemoryDocumentStore` remains for unit tests that inject it directly.

## Commands

```bash
# Seed / re-ingest the sample corpus into the configured store
.venv/bin/python -m scripts.ingest

# Optional durable backend
docker compose --profile vector-db up -d postgres
VECTOR_STORE=pgvector .venv/bin/python -m scripts.ingest
```

## Out of scope

- Chunking strategy bake-off (Step 7)
- Retrieval-grounded answers / citation validation (Step 8)
- Bounded concurrent ingestion jobs (Step 9)
