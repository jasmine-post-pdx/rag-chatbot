# Architecture

This page separates the system that runs today from the RAG architecture the
project is building toward. The current API and SQLite job store are working;
document ingestion and retrieval are still future work. Generation can use a
local placeholder or the configured OpenAI provider.

## Current implementation

```text
client
  -> FastAPI routes (`app/routers/`) and request middleware (`app/middleware.py`)
    -> question routes -> `QuestionWorkflow` (`app/services/question_workflow.py`)
      -> `QuestionProcessor` (`app/services/question_processor.py`)
      -> `QuestionJobStore` (`app/stores/question_job_store.py`)
        -> `SQLiteQuestionJobStore` (`app/stores/sqlite.py`)
    -> readiness -> `QuestionJobStore.is_ready()`
```

| Area | Current responsibility |
| --- | --- |
| `app/main.py`, `app/middleware.py` | Application lifespan, error handling, and request correlation. |
| `app/routers/`, `app/dependencies/` | Health, readiness, and question endpoints. Question routes use the workflow; readiness uses the job store. |
| `app/config/` | Environment-based settings and standard-library logging configuration. |
| `app/services/question_workflow.py` | Generates job IDs, coordinates answer processing, idempotency, and success or failure persistence. |
| `app/services/question_processor.py` | Processor port, placeholder and LLM implementations, and provider selection. |
| `app/stores/` | `QuestionJobStore` protocol and SQLite-backed implementation. |
| `scripts/question_job_maintenance.py` | SQLite snapshots, tiered backup rotation, and terminal-job archiving. |
| `app/domain/` | Internal job and answer DTOs, plus domain errors such as idempotency-key reuse, incomplete idempotency records, and invalid keys. |
| `app/schemas/` | Independent public question, health, and error schemas, including safe processing failure categories. |
| `app/llm/` | Generation interfaces, local fake, and OpenAI Responses adapter. |
| `app/retrieval/`, `app/tools/` | Package placeholders; no retrieval or tool behavior is implemented yet. |

The FastAPI application lifespan creates, verifies, and owns the SQLite store,
injects it into the workflow, and closes it on shutdown or failed startup after
creation. Question routes receive the workflow through dependency injection, and the
readiness route checks the same store directly. Question submission assigns a job
ID, checks any idempotency key, processes the question,
then persists and returns its terminal result in one request; in-flight work is
not stored. The HTTP request ID is passed independently to the processor.
Processing failures are persisted as failed jobs with a safe error category; the
submit request still returns a generic server error. API handlers map service
results and their nested metadata to public schemas. The store serializes jobs as
JSON in SQLite, migrates legacy schemas (for example older `question_id` +
`job_json` tables and renamed `request_id` columns) when adding maintenance and
idempotency metadata, and may rebuild the table to enforce the paired
idempotency CHECK. Readiness uses
`SELECT question_id FROM question_jobs LIMIT 1`. The optional Compose pgvector
service is not connected to the application; when that profile is used, the app
container still sets `POSTGRES_HOST=postgres` for the future vector-store path.
The maintenance command keeps 30 days of jobs live, archives older terminal
jobs, and expires archived rows after the configured archive period.

`scripts/question_job_maintenance.py` backs up, archives, and checks the SQLite
job store. The repository does not yet contain `data/raw/`, `data/processed/`,
or an ingestion pipeline.

## Tests today

| Path | Current contents |
| --- | --- |
| `tests/unit/` | Configuration, logging, SQLite job storage, question processors, and generation adapters. |
| `tests/integration/` | HTTP API behavior, OpenAPI and schema contracts, lifespan, request logging, generation wiring, and an opt-in OpenAI smoke test. |
| `tests/evals/` | Placeholder package; no retrieval or answer-quality evaluations exist yet. |

## Future RAG plan

The intended request flow will add document ingestion and retrieval before
generation:

```text
documents -> ingestion/chunking -> vector store
client -> API -> service/orchestrator -> retrieval -> model generation -> answer
                                      \-> optional tools
```

- Add ingestion, chunking, embeddings, search, ranking, and context assembly
  under `app/retrieval/` and additional model adapters under `app/llm/`.
- Connect the optional Postgres/pgvector service for document chunks and
  embeddings; keep SQLite focused on question-job state unless that boundary
  changes deliberately.
- Replace the placeholder answer with retrieval-grounded generation. Decide
  whether processing needs a background worker after real pipeline latency is
  understood.
- Add retrieval and answer-quality evaluations under `tests/evals/`.

## Decisions

Record architectural choices in `docs/decisions/` so later experiments can
compare against a known baseline rather than reconstructing why a chunker,
index, or model was selected.
