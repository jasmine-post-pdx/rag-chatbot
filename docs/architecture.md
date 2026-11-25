# Architecture

This page separates the system that runs today from the RAG architecture the
project is building toward. The current API, SQLite job store, document tools,
ingestion, and retrieval-only search are working. Retrieval-grounded generation
(Step 8) is still future work.

## Current implementation

```text
client
  -> async FastAPI routes (`app/routers/`) and request middleware (`app/middleware.py`)
    -> question routes -> await `QuestionWorkflow` (`app/services/question_workflow.py`)
      -> await to_thread(`QuestionJobStore`) for an idempotency lookup, when a key is present
      -> await `QuestionProcessor` (`app/services/question_processor.py`)
        -> `allocate_context_budget` (`app/services/token_budget.py`) before generate
           (tool schemas counted when document tools are enabled)
        -> OpenAI path without a document store:
             await `generate_structured(AnswerV1)` (`app/llm/`)
        -> OpenAI path with a document store:
             bounded `ToolLoop` (`app/tools/`) via `generate_tool_turn`
             -> allow-listed `SafeToolExecutor` -> vector `DocumentStore` (`app/retrieval/`)
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
| `app/services/question_processor.py` | Processor port, placeholder and LLM implementations, and provider selection. |
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
client. The same vector store backs `POST /v1/search` and OpenAI document tools.
Blocking store I/O is offloaded with `asyncio.to_thread`. Default embeddings use
a deterministic fake provider so local runs need no torch; set
`EMBEDDING_PROVIDER=local` and install `.[embeddings]` for sentence-transformers.
Default `VECTOR_STORE=memory`; set `pgvector` and start Compose profile
`vector-db` for durable vectors. Re-index deletes by `source_id` before upsert so
changed sources do not leave stale chunks. Cosine scores are similarity, not
calibrated confidence. See `docs/decisions/embeddings-retrieval.md`.

Citations remain empty until Step 8 grounding, while `RetrievalMetrics` may
reflect tool activity. `ClassificationV1` exists for later filter/tool routing
but is not invoked on live questions yet.

## Tests today

| Path | Current contents |
| --- | --- |
| `tests/unit/` | Configuration, logging, SQLite jobs, processors, typed outputs, budgeting, embeddings, chunking, ingestion, vector store, tools, tool loop, async generation, event-loop isolation. |
| `tests/integration/` | HTTP API (including search), OpenAPI contracts, lifespan, request logging, generation wiring, opt-in OpenAI smoke and pgvector tests. |
| `tests/evals/` | Representative retrieval queries with automated source-id checks against the fake embedder. |

## Future RAG plan

```text
documents -> ingestion/chunking -> vector store   (done in Step 6)
client -> API -> retrieve -> budget -> grounded generate -> answer   (Step 8)
```

- Compare chunking strategies and document the winner (Step 7).
- Wire ranked retrieval into the question path with citation validation (Step 8).
- Add bounded concurrent ingestion if batch jobs need it (Step 9).

## Decisions

Record architectural choices in `docs/decisions/` so later experiments can
compare against a known baseline rather than reconstructing why a chunker,
index, or model was selected.
