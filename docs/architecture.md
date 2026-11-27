# Architecture

This page separates the system that runs today from the RAG architecture the
project is building toward. The current API, SQLite job store, document tools,
ingestion, retrieval-only search, and retrieval-grounded generation (Step 8)
are working.

## Current implementation

```text
client
  -> async FastAPI routes (`app/routers/`) and request middleware (`app/middleware.py`)
    -> question routes -> await `QuestionWorkflow` (`app/services/question_workflow.py`)
      -> await to_thread(`QuestionJobStore`) for an idempotency lookup, when a key is present
      -> await `QuestionProcessor` (`app/services/question_processor.py`)
        -> OpenAI + document store (default factory path):
             `GroundedQuestionProcessor`
             optional classify/filter → store search (to_thread) →
             `allocate_context_budget` → `generate_structured(AnswerV1)` →
             `validate_and_map_citations` (selected chunks only)
        -> OpenAI without a document store:
             `LLMQuestionProcessor` → await `generate_structured(AnswerV1)`
        -> Tool loop remains available when `LLMQuestionProcessor` is constructed
             with a store directly (Step 5 path; not the factory default)
        -> local provider: `PlaceholderQuestionProcessor` (+ optional budget check)
      -> await to_thread(`QuestionJobStore`) to persist the terminal job
        -> `SQLiteQuestionJobStore` (`app/stores/sqlite.py`)
    -> search routes -> await to_thread(`SearchService`) -> vector `DocumentStore`
    -> readiness -> await to_thread(job store + optional pgvector readiness)
```

| Area | Current responsibility |
| --- | --- |
| `app/main.py`, `app/middleware.py` | Application lifespan, error handling, and request correlation. |
| `app/routers/`, `app/dependencies/` | Health, readiness, questions, and search endpoints. |
| `app/config/` | Environment-based settings and standard-library logging configuration. |
| `app/services/question_workflow.py` | Generates job IDs, coordinates answer processing, idempotency, and success or failure persistence. |
| `app/services/question_processor.py` | Processor port, placeholder/LLM/grounded implementations, and provider selection. |
| `app/services/citations.py` | Selected-chunk citation ID validation and public `Citation` mapping. |
| `app/services/search.py` | Retrieval-only search orchestration over a vector document store. |
| `app/services/token_budget.py` | Token counters and deterministic context-budget allocation before generation. |
| `app/stores/` | `QuestionJobStore` protocol and SQLite-backed implementation. |
| `scripts/ingest.py` | CLI to ingest `data/raw` into the configured vector store. |
| `scripts/question_job_maintenance.py` | SQLite snapshots, tiered backup rotation, and terminal-job archiving. |
| `app/domain/` | Internal job, answer, context-budget, and typed model-output DTOs, plus domain errors. |
| `app/schemas/` | Independent public question, search, health, and error schemas. |
| `app/llm/` | Async generation interfaces plus `EmbeddingClient` (fake default, optional local). |
| `app/retrieval/` | Chunking, ingestion, memory/pgvector vector stores, and the Step 5 substring stand-in. |
| `app/tools/` | Allow-listed read-only document tools, safe executor, and bounded tool loop. |

The FastAPI application lifespan creates the SQLite job store, embedding client,
and vector store; seeds `data/raw` markdown when present; and when OpenAI
generation is configured with an API key also owns a shared async OpenAI SDK
client. The same vector store backs `POST /v1/search` and the grounded OpenAI
question path. Blocking store I/O is offloaded with `asyncio.to_thread`. Default
embeddings use a deterministic fake provider so local runs need no torch; set
`EMBEDDING_PROVIDER=local` and install `.[embeddings]` for sentence-transformers.
Default `VECTOR_STORE=memory`; set `pgvector` and start Compose profile
`vector-db` for durable vectors. Re-index deletes by `source_id` before upsert so
changed sources do not leave stale chunks. Cosine scores are similarity, not
calibrated confidence. See `docs/decisions/embeddings-retrieval.md` and
`docs/decisions/grounded-answers.md`.

Public `citations` are populated from selected evidence after ID validation.
`ClassificationV1` can filter grounded retrieval when a classifier is injected;
it is not required on every live request.

## Tests today

| Path | Current contents |
| --- | --- |
| `tests/unit/` | Configuration, logging, SQLite jobs, processors, citations, grounded answers (incl. security/performance), typed outputs, budgeting, embeddings, chunking, ingestion (incl. bounded concurrent), vector store, tools, tool loop, async generation, event-loop isolation. |
| `tests/integration/` | HTTP API (including search and grounded questions), OpenAPI contracts, lifespan, request logging, generation wiring, opt-in OpenAI smoke and pgvector tests. |
| `tests/evals/` | Representative retrieval queries with automated source-id checks against the fake embedder; chunking experiment checks. |

## Future RAG plan

```text
documents -> bounded ingest/chunking -> vector store   (done in Step 6–7, 9)
client -> API -> retrieve -> budget -> grounded generate -> answer   (done in Step 8)
```

- CLI ingest (`python -m scripts.ingest`) is a separate process from the API, so
  long jobs stay isolated from question latency. Lifespan seeding uses the same
  bounded async pipeline with `INGESTION_CONCURRENCY` (1–10).

## Decisions

Record architectural choices in `docs/decisions/` so later experiments can
compare against a known baseline rather than reconstructing why a chunker,
index, or model was selected.
