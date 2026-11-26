# rag-chatbot — RAG Engineering Lab

> Status: This is an active learning project. The HTTP API, request/response
> schemas, SQLite-backed question-job storage, optional OpenAI generation,
> deterministic token/context budgeting, document tools, embeddings retrieval
> (`POST /v1/search`, ingestion, in-memory or pgvector stores), chunking
> experiments, and Step 8 retrieval-grounded answers with citation validation
> are implemented. Bounded concurrent ingestion (Step 9) is still ahead.

## Why this exists

This repository builds on my experience gained from my previous project localRAG,
focusing on refining existing skills and tackling more complex engineering challenges. 
It expands on the fundamentals of ingestion, chunking, embedding, retrieval, and 
generation by exploring more sophisticated pipeline design, improving retrieval quality, 
and handling the interactions between system components. The project also deepens 
my experience with API design, service-layer testing, and architectural decisions, 
emphasizing modularity, maintainability, and systematic evaluation as steps toward 
building more robust, production-ready AI applications.

## Project Structure

The application is organized so HTTP handling, orchestration, persistence, and
generation can change independently. Request flow today is:

```text
client
  -> async FastAPI routes (`app/routers/`) and request middleware (`app/middleware.py`)
    -> question routes -> await `QuestionWorkflow` (`app/services/question_workflow.py`)
      -> await to_thread(`QuestionJobStore`) for an idempotency lookup, when a key is present
      -> await `QuestionProcessor` (`app/services/question_processor.py`)
        -> OpenAI + store: GroundedQuestionProcessor
             retrieve → budget → structured AnswerV1 → validate citations
        -> otherwise: placeholder or structured generate without retrieval
      -> await to_thread(`QuestionJobStore`) to persist the terminal job
        -> `SQLiteQuestionJobStore` (`app/stores/sqlite.py`)
    -> search routes -> await to_thread(`SearchService`) -> vector `DocumentStore`
    -> readiness -> await to_thread(`QuestionJobStore.is_ready()`)
```

| Path | Role |
| --- | --- |
| `app/main.py` | Application entry point: lifespan, middleware, and error handlers. |
| `app/middleware.py` | Request correlation IDs and one-line request logging. |
| `app/routers/` | HTTP endpoints for health, readiness, questions, and search. |
| `app/dependencies/` | Injects the shared workflow, search service, and job store. |
| `app/schemas/` | Public request/response models for the HTTP contract. |
| `app/domain/` | Internal question/job DTOs, context-budget models, and domain errors. |
| `app/config/` | Environment-based settings and logging configuration. |
| `app/services/` | Question workflow, answer processors, citations, search, and token/context budgeting. |
| `app/stores/` | `QuestionJobStore` protocol and SQLite implementation. |
| `app/llm/` | Generation adapters plus `EmbeddingClient` (fake default, optional local). |
| `app/retrieval/` | Chunking, ingestion, vector `DocumentStore` (memory/pgvector), Step 5 substring stand-in. |
| `app/tools/` | Allow-listed read-only document tools, safe executor, and bounded tool loop. |
| `scripts/` | Ingestion CLI plus SQLite backup/archive maintenance. |
| `tests/unit/`, `tests/integration/` | Automated coverage for config, storage, processing, generation, retrieval, and the HTTP API. |
| `tests/evals/` | Representative retrieval queries and automated source-id checks. |
| `docs/` | Architecture notes, decisions, and API sequence diagrams. |

API handlers stay thin: they validate input, call services, and map results to
public schemas. Generation supports `local` (placeholder) and `openai`
adapters. With OpenAI and a document store, questions use
`GroundedQuestionProcessor` against the same vector store as `POST /v1/search`.
The Step 5 tool loop remains available via direct construction. Processing
failures are stored as failed jobs with `error_category: "processing_error"`;
submission still returns a generic HTTP 500. More detail lives in
[`docs/architecture.md`](docs/architecture.md),
[`docs/decisions/grounded-answers.md`](docs/decisions/grounded-answers.md),
[`docs/decisions/embeddings-retrieval.md`](docs/decisions/embeddings-retrieval.md),
and [`docs/decisions/tools.md`](docs/decisions/tools.md).

## Stack

- **Python 3.11+** — the project language.
- **FastAPI** — the web framework for the question API.
- **Pydantic** — request/response schemas and job models.
- **tiktoken** — token counting for OpenAI-oriented context budgeting.
- **Uvicorn** — ASGI server for running the API.
- **SQLite** — lightweight, local persistence for question jobs.
- **pytest** — test runner for unit, integration, and evaluation tests.
- **Ruff** — linter and formatter.
- **Docker + Docker Compose** — can run the API with persistent SQLite storage;
  optional Postgres/pgvector via the `vector-db` profile.

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

Key settings include the SQLite job-store path (`QUESTIONS_DB_PATH`), archive
and backup directories (`QUESTIONS_ARCHIVE_DIR`, `QUESTIONS_BACKUP_DIR`), live
and archive retention in days, the Postgres/pgvector connection details, and the
application log level (`LOG_LEVEL`, default `INFO`). `GENERATION_PROVIDER`
selects how questions are answered. `local` is the default and the stand-in
for a local model: it returns a placeholder answer and does not call a
provider. `openai` sends the question to `GENERATION_MODEL` through the
Responses API, without retrieval. Any other value fails at startup.
`OPENAI_API_KEY` is loaded through Pydantic Settings from the environment or
`.env` file, stored as a secret setting, and has no credential default. If
supplied but empty or whitespace-only, settings validation fails during
startup. The app can start without the variable, but an OpenAI-backed
question fails until a key is configured. Add this to `.env` when using
`GENERATION_PROVIDER=openai`:

```dotenv
OPENAI_API_KEY=your-key-here
```

Do not commit real credentials to the repository.

Local runs load all of these settings from environment variables or `.env`.
To use the local placeholder instead of OpenAI, set `GENERATION_PROVIDER=local`
or leave it unset.

Generation uses an async OpenAI Responses client (`AsyncOpenAI`). When
`GENERATION_PROVIDER=openai` and `OPENAI_API_KEY` is set, the FastAPI lifespan
creates a shared SDK client, injects it into the processor, and closes it on
shutdown. Free-text callers can await `generate`; the question processor awaits
`generate_structured` with versioned `AnswerV1` (Responses `parse` plus local
validation). Retry backoff uses non-blocking sleep so the event loop can
continue other work during waits.

The OpenAI client timeout is `GENERATION_TIMEOUT_SECONDS` (default `30`).
That limit applies to each attempt. Retries and backoff can make the total
wait longer than one timeout. `GENERATION_MAX_OUTPUT_TOKENS` defaults to
`1024` and is always sent as `max_output_tokens`. Override it to change the
cap. That same value is reserved from the model context window when allocating
input budget. `GENERATION_CONTEXT_WINDOW_TOKENS` (default `128000`) and
`GENERATION_BUDGET_SAFETY_MARGIN_TOKENS` (default `256`) control how much room
remains for retrieved evidence once instructions, the question, history, tool
schemas, framing, the output reserve, and the safety margin are subtracted.
OpenAI-oriented models count tokens with tiktoken; the local provider uses a
character-based estimate and labels budget results as estimates. Ranked chunks
are selected greedily in order; any chunk that does not fit the remaining
capacity is skipped so later smaller chunks can still be included. If fixed
input alone exceeds the window, the processor raises a controlled error and
does not call the provider. Budget metadata is logged without question,
history, tool, or chunk text; set `GENERATION_BUDGET_DEBUG=true` for extra
numeric fields. On the grounded OpenAI path, ranked retrieval hits are passed
into the budget allocator so only fitting chunks appear in the evidence prompt
and become citeable. `GENERATION_INSTRUCTIONS`, when set, is sent separately
from the question and counted as fixed input. Leave it unset to send no extra
instructions.

Structured outputs use internal schema version `"1"`
(`ClassificationV1` / `AnswerV1` in `app/domain/model_outputs.py`). The public
HTTP contract returns `answer` plus `citations` populated from selected
evidence after ID validation. Claims and `insufficient_evidence` stay
internal. Extra fields and unknown enums are rejected. When evidence is claimed
insufficient, claims must be empty; otherwise each claim needs at least one
citation chunk ID that appears in the selected set. Optional classification can
supply framework/language filters for grounded retrieval. See
`docs/decisions/typed-outputs.md` and `docs/decisions/grounded-answers.md`.

Structured recovery: provider refusals and local validation failures fail
closed as processing errors without inventing fields. An incomplete structured
finish gets one bounded re-request, then the same controlled failure. Prompt
and answer text are omitted from those logs.

The shared async SDK client is created with `max_retries=0`, so the SDK does
not retry on its own. The application retries HTTP 429, 500, 502, 503, and 504 up to
`GENERATION_MAX_RETRIES` (default `2`) extra attempts. Timeouts and connection
failures are retried at most once, even when that setting is higher, and no
retry starts after a 60-second wall-clock budget. Authentication failures and
bad requests are not retried. The wait before each retry is exponential
backoff with equal jitter: the wait is between half and all of a ceiling that
starts at half a second and caps at eight seconds. Each extra attempt is a new
Responses API create or parse. Those calls are not idempotent, so a transport
retry can bill another generation when an earlier request reached the provider;
those retries set `possible_duplicate_generation` on the warning log record
(along with `retry_class` and `elapsed_seconds`). The default stdout formatter
prints only a fixed field list, so those attributes appear in structured log
capture but not in the one-line request-style output unless that list is
extended. `input_tokens` and `output_tokens` are null when that provider omits
usage. The example below shows the local placeholder, which reports zero
because it does not call a model.

Docker Compose sets `QUESTIONS_DB_PATH` to `/app/data/questions.db` and forwards
the archive directory, backup directory, live and archive retention, and most
Postgres settings from the environment, using the same defaults as the
application. The app service always sets `POSTGRES_HOST=postgres` so it can
reach the optional Compose Postgres service; that value is not taken from
`.env` or the host environment. It also forwards `GENERATION_PROVIDER`,
`GENERATION_MODEL`, `GENERATION_TIMEOUT_SECONDS`, `GENERATION_MAX_RETRIES`,
`GENERATION_CONTEXT_WINDOW_TOKENS`, `GENERATION_BUDGET_SAFETY_MARGIN_TOKENS`,
and `GENERATION_BUDGET_DEBUG`, again using the application defaults when those
variables are unset. When a `.env` file is present, Compose loads it into the
app container. That is how `OPENAI_API_KEY`, `GENERATION_MAX_OUTPUT_TOKENS`, and
`GENERATION_INSTRUCTIONS` are forwarded without turning an unset value into an
empty string.
`LOG_LEVEL`, `EMBEDDING_PROVIDER`, and `EMBEDDING_MODEL` reach the container
only when they are set in that `.env` file; otherwise they stay at application
defaults. Exporting them in the host shell alone does not pass them through.

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
uv run uvicorn app.main:app --reload
```

With a pip-managed virtual environment:

```bash
uvicorn app.main:app --reload
```

### API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Report that the process is running. |
| `GET` | `/ready` | Report that the API is ready to accept requests. |
| `POST` | `/v1/questions` | Submit a question and get a completed result. |
| `GET` | `/v1/questions/{question_id}` | Retrieve the status and result of a submitted question. |

Submit a question:

```bash
curl -X POST http://localhost:8000/v1/questions \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Response:

```json
{
  "question_id": "<uuid>",
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

Finish status, refusal text, and the provider response ID stay off this
response. Status and the provider response ID are written to the generation
log; the question and refusal text are not. A refusal with no output text is
returned as the answer.

Retrieve the stored job:

```bash
curl http://localhost:8000/v1/questions/<question_id>
```

Clients can send an optional `Idempotency-Key` header to reuse a saved terminal
result after retrying the same question. Reusing a key with different question
text returns `409 Conflict`. The API does not persist an in-flight key claim, so
simultaneous requests with the same new key can both call the provider before
one terminal result is saved.

### Job retention, archives, and backups

The default policy keeps terminal jobs in the live database for 30 days. Their
idempotency keys are valid for at least that long; because archival runs daily,
a key can remain valid for up to one extra day. After archival deletes a job
from the live database, that key is no longer honored and can be reused for a
new submission. Jobs older than the live window move into monthly SQLite
archives for another 365 days, then expire. Archived jobs are kept for recovery
and are not returned by the normal status endpoint.
That is about 13 months in the live database and archives. Existing rows get a
`created_at` timestamp on the first upgraded app startup, so their retention
period starts at that migration time.

Run the maintenance commands from the repository root:

```bash
uv run python -m scripts.question_job_maintenance backup-live
uv run python -m scripts.question_job_maintenance archive --dry-run
uv run python -m scripts.question_job_maintenance archive
uv run python -m scripts.question_job_maintenance backup-archives
uv run python -m scripts.question_job_maintenance restore-check
```

`archive` first makes a live database snapshot, then copies and verifies old
terminal rows before deleting them from the live database. It is safe to rerun
after interruption. `--vacuum-live` can return freed live-database pages to the
operating system, but may take time and block database activity. Without it,
SQLite reuses those pages for later jobs. A dry run reports old nonterminal
records left in the live database for review; the current workflow creates only
terminal records.

Schedule `backup-live` hourly and `archive`, followed by `backup-archives`, once
a day. Backup snapshots use SQLite's backup API and retain all hourly snapshots
for 24 hours, daily snapshots for 30 days, weekly snapshots for 12 weeks, and
monthly snapshots for 365 days. `backup-archives` captures monthly archive
databases after the daily archival run. For a local app, a practical schedule
is:

```cron
0 * * * * cd /path/to/rag-chatbot && uv run python -m scripts.question_job_maintenance backup-live
15 2 * * * cd /path/to/rag-chatbot && uv run python -m scripts.question_job_maintenance archive
30 2 * * * cd /path/to/rag-chatbot && uv run python -m scripts.question_job_maintenance backup-archives
```

When the app runs in Compose, run maintenance in a one-off app container so it
uses the same SQLite volume as the server. Use these commands in place of the
`uv run` commands above:

```bash
docker compose run --rm --no-deps app python -m scripts.question_job_maintenance backup-live
docker compose run --rm --no-deps app python -m scripts.question_job_maintenance archive
docker compose run --rm --no-deps app python -m scripts.question_job_maintenance backup-archives
```

By default, archives and backups live under `data/` alongside the live database.
For recovery from disk or host loss, set `QUESTIONS_BACKUP_DIR` to encrypted,
off-host storage. In Compose, mount that same path into the app container so the
server and maintenance command can read and write it. Backup snapshots can
contain records that have since expired from the live database or archives until
those snapshots age out. An archived row near the end of its 395-day live plus
archive window can remain in an archive backup for nearly another year, or
about 25 months total. `restore-check` restores the newest live snapshot to a
temporary database, runs `quick_check`, and verifies the job table. Run it
periodically; also perform a full service recovery drill by restoring the live
and archive snapshots while the application is stopped, then run `archive`
before reopening traffic so expired idempotency keys are removed.

### Request logging

Every request is logged with its request ID, HTTP method, route, response
status, and elapsed time:

```
INFO:     Request completed request_id=<uuid> method=POST route=/v1/questions status=200 elapsed_ms=12.345
```

You can provide your own request ID in the `x-request-id` header; otherwise
the API generates one and returns it in the response:

```bash
curl -X POST http://localhost:8000/v1/questions \
  -H "Content-Type: application/json" \
  -H "x-request-id: my-correlation-id" \
  -d '{"question": "What is RAG?"}'
```

Request logs for validation failures (`422`) and server errors (`500`) include
an `error_category` field. Logs for other responses, including `404` and `409`,
record the status without that field. `error_category` is log metadata; HTTP
error responses return a concise `detail` field instead:

```
INFO:     Request completed request_id=<uuid> method=POST route=/v1/questions status=422 elapsed_ms=2.1 error_category=validation
INFO:     Request completed request_id=<uuid> method=POST route=/v1/questions status=500 elapsed_ms=5.7 error_category=server
```

If question processing fails after the job is saved, the generic `500` response
also includes an `x-question-id` header. That header is the `question_id`. Use
it to poll `GET /v1/questions/{question_id}`; the job response reports
`status: "failed"`
and `error_category: "processing_error"` without exposing provider exception
details. A corrupted or incomplete idempotency-key row returns `500` with
`detail: "Idempotency-Key record is incomplete"` and no `x-question-id`.

The middleware owns one request summary; API exception handlers own handled
error records. Error records include the request ID, category, exception type,
and a bounded one-line summary. Common credential patterns are redacted, and
tracebacks are not written to stdout. This redaction is best-effort: provider
adapters should avoid putting prompts, personal data, or secrets in exception
messages. Invalid or oversized client-supplied `x-request-id` values are
replaced with a generated ID before they are logged or returned.

### Local Postgres with pgvector

A Postgres container with pgvector is available when `VECTOR_STORE=pgvector`:

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

- Answers are placeholders by default (`GENERATION_PROVIDER=local`). With
  `GENERATION_PROVIDER=openai` and a seeded vector store, answers are
  retrieval-grounded and may include citations for selected chunks. Invented or
  budget-discarded citation IDs fail closed. Oversized fixed input still fails
  before a provider call.
- `POST /v1/search` is retrieval-only. Ingest sample markdown with
  `python -m scripts.ingest` (or rely on lifespan seeding of `data/raw`).
  Compare chunking strategies on an isolated in-memory index with
  `python -m scripts.chunking_experiment --path data/raw --output docs/evals/chunking-experiment-results.json`
  (never clears the live `documents` collection).
- `POST /v1/questions` still finishes in one request: it awaits generation,
  persists the completed or failed job, and returns. SQLite work is offloaded
  off the event loop. In-flight work is not persisted; the status endpoint
  retrieves terminal job records after submission finishes. A durable worker
  that survives disconnects or restarts is a future option, not part of this
  async conversion.
- Making generation async does not by itself impose application concurrency
  limits. Uvicorn worker count, OS resources, and provider rate limits still
  govern how much work runs at once. Per-attempt `GENERATION_TIMEOUT_SECONDS`
  plus retries and backoff can make total request latency longer than one
  timeout.
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

The default `pytest` command excludes the `openai_smoke`, `embeddings_smoke`,
and `vector_db` markers. Opt-in examples:

```bash
RUN_OPENAI_SMOKE=1 pytest -m openai_smoke
RUN_VECTOR_DB=1 pytest -m vector_db
```

The OpenAI smoke test makes a real provider request and may incur API usage
charges. The vector_db tests need a reachable Postgres/pgvector instance.

Run the tests inside the Docker `dev` target:

```bash
docker build --target dev -t rag-chatbot:dev .
docker run --rm rag-chatbot:dev
```

Run the linter:

```bash
ruff check app tests
```

The integration tests exercise the HTTP API through a dependency override
backed by an in-memory SQLite store. Unit tests cover configuration, logging,
job storage, processors (including grounded answers and citation validation),
token/context budgeting, document tools and the bounded tool loop, the async
generation adapters, and event-loop isolation when SQLite or generation is
blocked or cancelled. `tests/evals/` holds representative retrieval queries;
broader answer-quality evals are still ahead.

## What's next

The following are proposed next steps. They will be implemented to the extent
that they support this project's learning goals. Some may be deferred to future
projects.

- [x] Add real document ingestion (load, chunk, embed) under `app/retrieval/`
      (Step 6).
- [x] Add an embedding model adapter under `app/llm/` (fake default; optional
      local sentence-transformers) (Step 6).
- [x] Store chunks and embeddings in memory by default, with optional
      Postgres/pgvector (Step 6).
- [x] Add retrieval-only `POST /v1/search` and representative retrieval queries
      under `tests/evals/` (Step 6).
- [x] Compare chunking strategies and record the decision (Step 7).
- [x] Add a generation adapter that builds prompts from retrieved context
      (Step 8).
- [x] Replace empty citations with retrieval-grounded generation and selected-
      chunk citation validation (Step 8).
- [x] Move database path and model settings into configuration or environment
      variables.
- [x] Await generation asynchronously and offload blocking SQLite work from the
      event loop (Step 2.5).
- [x] Add deterministic token/context budgeting before retrieval is wired
      (Step 3).
- [x] Add allow-listed read-only document tools and a bounded tool loop
      (Step 5).
- [ ] Add bounded concurrent ingestion for genuine async I/O (Step 9).
- [ ] Make question jobs durable across disconnects or restarts with a real
      pending/completed worker lifecycle when latency requires it.

## License

This project is licensed under the MIT license.
