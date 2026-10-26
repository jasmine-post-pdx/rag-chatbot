# Architecture

This page separates the system that runs today from the RAG architecture the
project is building toward. The current API and SQLite job store are working;
document ingestion, retrieval, and model integrations are still future work.

## Current implementation

```text
client
  -> FastAPI routes and request middleware (`app/api/`)
    -> `QuestionWorkflow` (`app/services/question_workflow.py`)
      -> `QuestionProcessor` (`app/services/question_processor.py`)
      -> `QuestionJobStore` (`app/stores/question_jobs.py`)
        -> `SQLiteQuestionJobStore` (`app/stores/sqlite_question_jobs.py`)
```

| Area | Current responsibility |
| --- | --- |
| `app/api/` | Health and readiness checks, synchronous question submission, job status responses, validation, and request correlation. |
| `app/core/` | Environment-based settings and standard-library logging configuration. |
| `app/services/question_workflow.py` | Coordinates job creation, answer processing, and success or failure persistence. |
| `app/services/question_processor.py` | Processor interface and the current placeholder answer implementation. |
| `app/stores/` | Job-store interface and SQLite-backed implementation. |
| `app/models/questions.py` | Internal job and answer DTOs shared by processing, storage, and API layers. |
| `app/api/schemas.py` | Public response model for job status, including safe processing failure categories. |
| `app/retrieval/`, `app/tools/` | Package placeholders; no retrieval or tool behavior is implemented yet. |

The workflow and SQLite store are initialized during the FastAPI application
lifespan and shared with routes through dependency injection. Question
submission creates, processes, and returns a job in one request. Processing
failures are persisted as failed jobs with a safe error category; the submit
request still returns a generic server error. SQLite stores the job;
the optional Compose pgvector service is not connected to the application.

The `data/raw/`, `data/processed/`, and `scripts/` directories are reserved for
future ingestion work and currently contain no pipeline implementation.

## Tests today

| Path | Current contents |
| --- | --- |
| `tests/unit/` | Configuration and SQLite question-service tests. |
| `tests/integration/` | API behavior, request handling, and service failure cases. |
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
  under `app/retrieval/` and model adapters under `app/models/`.
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
