# Architecture

This page separates the system that runs today from the RAG architecture the
project is building toward. The current API and SQLite job store are working;
document ingestion and retrieval are still future work. Generation can use a
local placeholder or the configured OpenAI provider.

## Current implementation

```text
client
  -> async FastAPI routes (`app/routers/`) and request middleware (`app/middleware.py`)
    -> question routes -> await `QuestionWorkflow` (`app/services/question_workflow.py`)
      -> await to_thread(`QuestionJobStore`) for an idempotency lookup, when a key is present
      -> await `QuestionProcessor` (`app/services/question_processor.py`)
        -> `allocate_context_budget` (`app/services/token_budget.py`) before generate
        -> await `generate_structured(AnswerV1)` (`app/llm/`) for the OpenAI path
      -> await to_thread(`QuestionJobStore`) to persist the terminal job
        -> `SQLiteQuestionJobStore` (`app/stores/sqlite.py`)
    -> readiness -> await to_thread(`QuestionJobStore.is_ready()`)
```

| Area | Current responsibility |
| --- | --- |
| `app/main.py`, `app/middleware.py` | Application lifespan, error handling, and request correlation. |
| `app/routers/`, `app/dependencies/` | Health, readiness, and question endpoints. Question routes use the workflow; readiness uses the job store. |
| `app/config/` | Environment-based settings and standard-library logging configuration. |
| `app/services/question_workflow.py` | Generates job IDs, coordinates answer processing, idempotency, and success or failure persistence. |
| `app/services/question_processor.py` | Processor port, placeholder and LLM implementations, and provider selection. |
| `app/services/token_budget.py` | Token counters and deterministic context-budget allocation before generation. |
| `app/stores/` | `QuestionJobStore` protocol and SQLite-backed implementation. |
| `scripts/question_job_maintenance.py` | SQLite snapshots, tiered backup rotation, and terminal-job archiving. |
| `app/domain/` | Internal job, answer, context-budget, and typed model-output DTOs, plus domain errors such as idempotency-key reuse, incomplete idempotency records, invalid keys, context-budget overflow, and structured-output failures. |
| `app/schemas/` | Independent public question, health, and error schemas, including safe processing failure categories. |
| `app/llm/` | Async free-text and schema-constrained generation interfaces, local fake, and OpenAI Responses adapter. |
| `app/retrieval/`, `app/tools/` | Package placeholders; no retrieval or tool behavior is implemented yet. |

The FastAPI application lifespan creates, verifies, and owns the SQLite store,
and when OpenAI generation is configured with an API key it also owns a shared
async OpenAI SDK client. Both are injected into the workflow path and closed on
shutdown or failed startup after creation. Question routes receive the workflow through dependency injection, and the
readiness route checks the same store directly. `QuestionWorkflow` awaits the
async processor and offloads each discrete SQLite call with `asyncio.to_thread`,
so blocking DB I/O does not run on the event loop and no store lock is held
across a provider await. Question submission assigns a job
ID, checks any idempotency key, processes the question,
then persists and returns its terminal result in one request; in-flight work is
not stored. Async generation alone does not add a durable background worker, and it does
not impose application concurrency limits by itself. Per-attempt generation
timeouts plus retries can make total latency longer than one timeout.
The HTTP request ID is passed independently to the processor.
Before generation, processors allocate a context budget: the model context
window minus instructions, question, history, tool schemas, request framing,
the output-token reserve, and a safety margin. OpenAI-oriented models use
tiktoken; the local path uses a character-based estimate and labels results as
estimates. Ranked chunks are selected greedily in order; chunks that do not fit
are skipped. If fixed input alone exceeds the window, processing fails closed
without calling the provider. Live requests still supply an empty chunk list
because retrieval is not wired yet. Budget logs omit prompt and chunk text.
The OpenAI processor path then requests schema-constrained `AnswerV1` output
(`generate_structured`), re-validates locally, and maps only the answer text
onto the public result; citations remain empty until retrieval grounding.
`ClassificationV1` exists for later filter/tool steps but is not invoked on
live questions yet. Incomplete structured finishes get one bounded re-request;
refusals and validation failures become controlled processing failures. See
`docs/decisions/typed-outputs.md`.
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
| `tests/unit/` | Configuration, logging, SQLite job storage, question processors, typed model-output schemas, token/context budgeting, async generation adapters, and event-loop isolation. |
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

- Add ingestion, chunking, embeddings, search, and ranking under
  `app/retrieval/` and additional model adapters under `app/llm/`. Context
  budgeting already lives in `app/services/token_budget.py` and will consume
  ranked chunks once retrieval exists.
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
