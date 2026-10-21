# rag-chatbot — RAG Engineering Lab

> Status: This is an active learning project. The HTTP API, request/response
> schemas, and SQLite-backed question-job storage are implemented, but the
> retrieval and generation stages still return placeholders. I consider the
> current milestone complete enough to build on: the structure is testable,
> documented, and ready for real RAG adapters to be dropped in.

## Why this exists

This repository builds on my experience gained from my previous project localRAG,
focusing on refining existing skills and tackling more complex engineering challenges. 
It expands on the fundamentals of ingestion, chunking, embedding, retrieval, and 
generation by exploring more sophisticated pipeline design, improving retrieval quality, 
and handling the interactions between system components. The project also deepens 
my experience with API design, service-layer testing, and architectural decisions, 
emphasizing modularity, maintainability, and systematic evaluation as steps toward 
building more robust, production-ready AI applications.

## From minimal scaffold to structured project

The project began as a single FastAPI endpoint that accepted a question and
returned a placeholder answer. I reorganized that into a more conventional
application structure so each part of the pipeline can evolve independently:

- `app/api/main.py` is the HTTP entry point. It defines the request and
  response schemas, wires the question service through FastAPI dependency
  injection, and maps errors to clean HTTP responses.
- `app/core/config.py` centralizes settings loaded from environment variables
  and an optional `.env` file. It keeps environment-specific values out of the
  source code.
- `app/services/questions.py` owns the question-job lifecycle: creating,
  completing, and looking up jobs. Jobs are persisted to SQLite so the API can
  survive restarts and support asynchronous polling.
- `app/retrieval/` will own chunking, indexing, search, ranking, and context
  assembly once the retrieval adapters are added.
- `app/models/` will hold LLM and embedding adapters when real generation is
  wired up.
- `app/tools/` is reserved for optional callable tools the generator can use.
- `tests/unit/`, `tests/integration/`, and `tests/evals/` mirror these
  responsibilities and hold the corresponding automated tests.

This separation is deliberate, not just organizational. Keeping API handlers
thin, the service layer stateful but storage-agnostic, and the retrieval and
model packages isolated makes it easier to replace the placeholder behavior
with real implementations without rewriting the whole application.

## What I'm learning here

- FastAPI request validation, dependency injection, and exception handling
- Pydantic models as API schemas and internal data-transfer objects
- Persisting short-lived jobs to SQLite with thread-safe access
- Writing testable services and covering them with unit and integration tests
- Structuring a RAG pipeline so retrieval, generation, and orchestration can
  evolve independently
- Recording architectural decisions as the project grows

## Stack

- **Python 3.11+** — the project language.
- **FastAPI** — the web framework for the question API.
- **Pydantic** — request/response schemas and job models.
- **Uvicorn** — ASGI server for running the API.
- **SQLite** — lightweight, local persistence for question jobs.
- **pytest** — test runner for unit, integration, and evaluation tests.
- **Ruff** — linter and formatter.
- **Docker Compose + Postgres/pgvector** — provided for future vector-store
  work; not connected to the API yet.

## Running it

### Prerequisites

- Python 3.11 or newer
- (Optional) Docker for running the Postgres/pgvector container

From the repository root:

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Configuration

The application reads settings from environment variables and an optional `.env`
file. Copy `.env.example` to `.env` and edit the values you need:

```bash
cp .env.example .env
```

Key settings include the SQLite job-store path (`QUESTIONS_DB_PATH`) and the
Postgres/pgvector connection details. All settings have local-development
defaults, so the API runs without a `.env` file.

### Run the API

With uv:

```bash
uv run uvicorn app.api.main:app --reload
```

With a pip-managed virtual environment:

```bash
uvicorn app.api.main:app --reload
```

### API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Report that the process is running. |
| `GET` | `/ready` | Report that the API is ready to accept requests. |
| `POST` | `/v1/questions` | Submit a question and get a completed result. |
| `GET` | `/v1/questions/{request_id}` | Poll the status of a submitted question. |

Submit a question:

```bash
curl -X POST http://localhost:8000/v1/questions \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Response:

```json
{
  "request_id": "<uuid>",
  "answer": "This is a placeholder answer for: What is RAG?",
  "citations": [],
  "retrieval": {"documents_considered": 0, "chunks_returned": 0},
  "model": {
    "provider": "local-or-api",
    "name": "stub",
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

Poll the job:

```bash
curl http://localhost:8000/v1/questions/<request_id>
```

### Local Postgres with pgvector

A Postgres container with pgvector is provided for future vector-store work:

```bash
docker compose up -d
```

It is not yet used by the application; the API currently stores question jobs
in `data/questions.db`.

## Data And Privacy

Question jobs are stored in a local SQLite file at `data/questions.db`. The file
is created at runtime and is ignored by git. The app does not write
question/answer logs or call external LLM APIs by default. The
Postgres container keeps its data in a named Docker volume and is isolated from
the SQLite store until the retrieval layer is wired up.

## Limitations And Troubleshooting

- Answers are placeholders. Retrieval and generation are not implemented yet.
- Only the question-job endpoints are functional; there is no document ingestion
  or search endpoint.
- `POST /v1/questions` is synchronous: it creates, completes, and returns the
  job in one request. Real RAG work will likely be asynchronous.
- The SQLite database path defaults to `data/questions.db` and can be changed
  with the `QUESTIONS_DB_PATH` environment variable or in a `.env` file.
- The Postgres container must be started manually and is not required to run
  the current API.
- If `data/` does not exist, the service creates it automatically on startup.

## Development

Run the automated tests from the project root:

```bash
pytest
```

Run the linter:

```bash
ruff check app tests
```

The integration tests exercise the HTTP API with an in-memory service override,
and the unit/evals directories are ready for retrieval, generation, and quality
tests as those layers are added.

## Notes & findings

- **2026-10-20:** A small, testable scaffold is easier to extend than a
  monolithic prototype. Separating the API, service, and storage layers made
  the integration tests simple to write without invoking real retrieval or
  generation.
- **2026-10-20:** Pydantic models work well as both request/response schemas
  and internal DTOs, but it is important to keep internal-only fields out of
  public response models as the job state grows.
- **2026-10-20:** SQLite with a thread lock is sufficient for a single-process
  FastAPI app using thread-pool workers, but it will not scale to multiple
  processes or workers.
- **2026-10-20:** Placeholder endpoints reveal what the API contract should be
  before the heavy RAG machinery exists: question in, answer + citations +
  retrieval/model metadata out.

## What's next

The following are proposed next steps. They will be implemented to the extent
that they support this project's learning goals. Some may be deferred to future
projects.

- [ ] Add real document ingestion (load, chunk, embed) under `app/retrieval/`.
- [ ] Add a local or API-based embedding model adapter under `app/models/`.
- [ ] Add a generation adapter that builds prompts from retrieved context.
- [ ] Replace the placeholder answer in `complete_job` with a real RAG chain.
- [ ] Store chunks and embeddings in the Postgres/pgvector container.
- [ ] Add retrieval-quality evaluations under `tests/evals/`.
- [x] Move database path and model settings into configuration or environment
      variables.
- [ ] Make question processing asynchronous with a real pending/completed
      lifecycle.

## License

This project is licensed under the MIT license.
