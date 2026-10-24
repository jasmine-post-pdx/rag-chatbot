# Architecture

This page separates the system that runs today from the RAG architecture the
project is building toward. The current API and SQLite job store are working;
document ingestion, retrieval, and model integrations are still future work.

## Current implementation

```text
client
  -> FastAPI routes and request middleware (`app/api/`)
    -> shared `QuestionService` (`app/services/questions.py`)
      -> SQLite question-job database
```

| Area | Current responsibility |
| --- | --- |
| `app/api/` | Health and readiness checks, synchronous question submission, job status responses, validation, and request correlation. |
| `app/core/` | Environment-based settings and standard-library logging configuration. |
| `app/services/questions.py` | Question-job lifecycle, SQLite persistence, and the placeholder answer. Storage and orchestration currently live together. |
| `app/api/schemas.py` | Public response model for job status. Other API and result models remain in `main.py` and `services/questions.py`. |
| `app/models/`, `app/retrieval/`, `app/tools/` | Package placeholders; no model, retrieval, or tool behavior is implemented yet. |

The service is initialized during the FastAPI application lifespan and shared
with routes through dependency injection. Question submission currently
creates, completes, and returns a job in one request. SQLite stores the job;
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
- Consider separating job orchestration from SQLite persistence when another
  job store or a different deployment model is needed.

## Decisions

Record architectural choices in `docs/decisions/` so later experiments can
compare against a known baseline rather than reconstructing why a chunker,
index, or model was selected.
