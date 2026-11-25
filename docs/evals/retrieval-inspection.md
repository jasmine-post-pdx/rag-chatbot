# Retrieval inspection (Step 6)

Representative queries live in
[`tests/evals/retrieval_queries.json`](../../tests/evals/retrieval_queries.json)
(30 non-secret questions against the sample corpus).

## Automated run (completed)

Command:

```bash
.venv/bin/pytest tests/evals/test_retrieval_queries.py -q
```

Configuration actually used:

| Field | Value |
| --- | --- |
| Corpus | In-code mirror of `data/raw/*.md` (fastapi-guide, fastapi-lifespan, rag-budget, express-guide) |
| Embedding provider | `FakeEmbeddingClient` (`fake-embed`, 32 dims) |
| Vector store | `InMemoryVectorDocumentStore` (cosine) |
| top_k | 5 |

Result: every fixture query returned at least one expected `source_id` in top-k
under the fake embedder. This is a deterministic stand-in for manual inspection,
not proof of semantic quality with a production embedding model.

## Not run in this step

| Check | Why |
| --- | --- |
| Live `sentence-transformers/all-MiniLM-L6-v2` | Optional `[embeddings]` extra; not required for the default suite |
| Live Postgres/pgvector ranking of the 30 queries | Opt-in (`RUN_VECTOR_DB=1`); default CI uses the memory store |
| Human graded relevance beyond source_id match | Deferred; source-id hit rate is the recorded automated signal |

To re-check with a local model after installing `.[embeddings]`:

```bash
EMBEDDING_PROVIDER=local \
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
.venv/bin/python -m scripts.ingest
```

Then call `POST /v1/search` for each query in the JSON fixture and record hits
here.
