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

- `app/api/main.py` is the HTTP entry point. It wires the question service
  through FastAPI dependency injection and maps errors to clean HTTP responses.
- `app/api/schemas.py` currently holds the public polling response schema.
  Other request and response models still live in the API and service modules;
  moving all public schemas here is a future refinement.
- `app/core/config.py` centralizes settings loaded from environment variables
  and an optional `.env` file. It keeps environment-specific values out of the
  source code.
- `app/services/questions.py` currently implements both the question-job
  lifecycle and SQLite persistence. Completed jobs survive restarts and can be
  retrieved through the status endpoint. A future storage abstraction could
  separate orchestration from SQLite and support other job stores.
- `app/retrieval/` will own chunking, indexing, search, ranking, and context
  assembly once the retrieval adapters are added.
- `app/models/` will hold LLM and embedding adapters when real generation is
  wired up.
- `app/tools/` is reserved for optional callable tools the generator can use.
- `tests/unit/`, `tests/integration/`, and `tests/evals/` mirror these
  responsibilities and hold the corresponding automated tests.

This structure establishes boundaries that will be strengthened as the RAG
pipeline develops. API handlers are kept thin, while the current question
service handles both job orchestration and SQLite persistence. Retrieval,
model, and tool packages are placeholders for future adapters. Public API
schemas are partially separated from internal models, with further separation
planned as the contract grows.

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
- **Docker + Docker Compose** — can run the API with persistent SQLite storage;
  Postgres/pgvector remains optional future vector-store work.

## Running it

### Prerequisites

- Python 3.11 or newer
- `uv` when using the recommended `uv sync` workflow
- (Optional) Docker with the Compose plugin for containerized development

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

Key settings include the SQLite job-store path (`QUESTIONS_DB_PATH`), the
Postgres/pgvector connection details, and the application log level
(`LOG_LEVEL`, default `INFO`). All settings have local-development defaults, so
the API runs without a `.env` file.

Local runs load all of these settings from environment variables or `.env`.
Docker Compose currently forwards the Postgres settings, sets
`QUESTIONS_DB_PATH` to `/app/data/questions.db`, and otherwise uses application
defaults. Settings such as `LOG_LEVEL` and model configuration will need to be
added to the Compose environment before `.env` overrides affect them inside the
container.

### Run with Docker

Build the API image (default `runtime` target) and start the API. The port
mapping is defined in `docker-compose.yml`, so no extra `-p` flag is needed:

```bash
docker compose up --build
```

The API is available at <http://localhost:8000>. Compose stores the SQLite job
database in a named volume, so it survives container restarts and image
rebuilds. Stop the service with `Ctrl+C`, or run `docker compose down`; either
keeps the database volume. To remove the stored jobs as well, run
`docker compose down --volumes`.

To start the optional pgvector service alongside the API:

```bash
docker compose --profile vector-db up --build
```

The API does not connect to Postgres yet, so this profile is only useful for
future vector-store development.

### Run tests in Docker

The Dockerfile defaults to the production `runtime` target. To build the `dev`
target with test dependencies and run the test suite:

```bash
docker build --target dev -t rag-chatbot:dev .
docker run --rm rag-chatbot:dev
```

The `runtime` and `dev` targets use the same lockfile and runtime dependency
versions. The `dev` target additionally installs test and lint dependencies. A
CI workflow has not yet been added.

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
| `GET` | `/v1/questions/{request_id}` | Retrieve the status and result of a submitted question. |

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

Retrieve the stored job:

```bash
curl http://localhost:8000/v1/questions/<request_id>
```

### Request logging

Every request is logged with a correlation ID, HTTP method, route, response
status, and elapsed time:

```
INFO:     Request completed request_id=<uuid> method=POST route=/v1/questions status=200 elapsed_ms=12.345
```

You can provide your own correlation ID in the `x-request-id` header; otherwise
the API generates one and returns it in the response:

```bash
curl -X POST http://localhost:8000/v1/questions \
  -H "Content-Type: application/json" \
  -H "x-request-id: my-correlation-id" \
  -d '{"question": "What is RAG?"}'
```

Failure request logs include an `error_category` field. This is log metadata;
HTTP error responses return a concise `detail` field instead:

```
INFO:     Request completed request_id=<uuid> method=POST route=/v1/questions status=422 elapsed_ms=2.1 error_category=validation
INFO:     Request completed request_id=<uuid> method=POST route=/v1/questions status=500 elapsed_ms=5.7 error_category=server
```

Request logging does not intentionally include request bodies, answers,
credentials, or settings objects. Correlation and request metadata are emitted
as structured key/value fields using the standard library. Exception handlers
can include exception messages in log messages, but the configured stdout
formatter does not render traceback text from exception records. Sanitize
provider exception messages before logging them when external services are
integrated.

### Local Postgres with pgvector

A Postgres container with pgvector is provided for future vector-store work:

```bash
docker compose --profile vector-db up -d postgres
```

It is not yet used by the application. The API stores question jobs in SQLite,
at `data/questions.db` when run locally or in the Compose-managed `questions_data`
volume when run in Docker.

## Data And Privacy

Question jobs are stored in SQLite. Local runs use `data/questions.db`; Docker
runs use a named volume mounted at `/app/data`. The app does not write
question/answer logs or call external LLM APIs by default. The optional Postgres
container keeps its data in a separate named Docker volume and is isolated from
the SQLite store until the retrieval layer is wired up.

## Limitations And Troubleshooting

- Answers are placeholders. Retrieval and generation are not implemented yet.
- Only the question-job endpoints are functional; there is no document ingestion
  or search endpoint.
- `POST /v1/questions` is synchronous: it creates, completes, and returns the
  job in one request. Real RAG work will likely be asynchronous.
- The SQLite database path defaults to `data/questions.db` and can be changed
  with the `QUESTIONS_DB_PATH` environment variable or in a `.env` file.
- Postgres is optional and is not required to run the current API. Start it with
  the `vector-db` Compose profile when needed.
- If `data/` does not exist, the service creates it automatically on startup.

## Development

Run the automated tests from the project root:

```bash
pytest
```

Run the tests inside the Docker `dev` target:

```bash
docker build --target dev -t rag-chatbot:dev .
docker run --rm rag-chatbot:dev
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
  monolithic prototype. Separating API handlers from service logic made the
  integration tests simple to write without invoking real retrieval or generation.
- **2026-10-23:** Pydantic models work well as both request/response schemas
  and internal DTOs, but it is important to keep internal-only fields out of
  public response models as the job state grows. A dedicated `QuestionJobResponse`
  schema in `app/api/schemas.py` now maps the internal `QuestionJob` DTO to the
  public status contract.
- **2026-10-20:** SQLite with a thread lock is sufficient for a single-process
  FastAPI app using thread-pool workers, but it will not scale to multiple
  processes or workers.
- **2026-10-20:** Placeholder endpoints reveal what the API contract should be
  before the heavy RAG machinery exists: question in, answer + citations +
  retrieval/model metadata out.
- **2026-10-20:** FastAPI dependency overrides make service-level failure cases
  easy to test without patching module globals. Managing the shared service
  through the application lifespan also gives database connections an explicit
  startup and shutdown boundary.
- **2026-10-21:** Liveness and readiness answer different questions. `/health`
  reports that the process is running, while `/ready` verifies that SQLite is
  usable and returns `503` when the dependency is unavailable.
- **2026-10-21:** Stable error responses require testing malformed JSON,
  validation failures, invalid service output, and dependency failures—not
  only successful requests. Clients receive concise errors while correlation
  IDs preserve a path for diagnosis.
- **2026-10-22:** Runtime invariants should use explicit error handling rather
  than `assert`. Assertions can be disabled and do not provide a deliberate
  HTTP response when a service unexpectedly returns an incomplete result.
- **2026-10-22:** SQLite needs explicit concurrency and integrity decisions.
  Locking protects a shared connection within one process, WAL mode improves
  reader/writer behavior, configurable timeouts handle temporary contention,
  and `INSERT`/`UPDATE` preserve duplicate and missing-job errors that
  `REPLACE` could hide.
- **2026-10-22:** Configuration values often need transformation as well as
  validation. Database usernames, passwords, and names must be URL-encoded
  before being assembled into a connection string.
- **2026-10-23:** Multi-stage Docker builds let development and runtime images
  share locked runtime dependencies while keeping tests and development tools
  out of the production image. Running as a non-root user and persisting data
  through named volumes should be designed in from the start.
- **2026-10-23:** Observability is also a security boundary. Structured request
  metadata and correlation IDs are useful, but exception messages from future
  adapters must be sanitized before they are written to logs.

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
