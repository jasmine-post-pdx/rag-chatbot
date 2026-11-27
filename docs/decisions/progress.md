# Learning Progress

This file is a running log of what I figure out while learning AI tech — especially retrieval-augmented generation, embeddings, evaluation, and the systems around them.

The point is not a polished essay. Each entry should capture what clicked, why it matters, and what is still fuzzy so I can see the trail later.

Newest entries go at the top. Older entries describe the implementation and
next steps as they stood at the time, not necessarily the current state.

---

## 2026-11-27 — Bounded async ingestion

**What I figured out**
Step 9 adds `BoundedIngestionPipeline`: semaphore-capped concurrent
`aembed_documents` calls (1–10), `to_thread` for parse/chunk/sync store I/O,
per-source and job deadlines, cancel-safe cleanup, and partial-failure
reporting via `ConcurrentIngestResult`. The CLI and lifespan seed share that
path; embedding clients gained `aembed_*` / `aclose`. No durable ingest queue.

**Why it matters**
Batch ingest can use real concurrency against provider limits without freezing
the API event loop, and CLI ingest stays in a separate process from questions.

**Still fuzzy**
- When a durable ingest worker would beat process-isolated CLI for demos
- Whether pgvector should grow a native async driver instead of `to_thread`

**Next**
Phase One wrap-up (architecture/docs already largely present); durable question
workers only if latency demands them.

See also `docs/decisions/bounded-ingestion.md`.

---

## 2026-11-26 — Retrieval-grounded answers

**What I figured out**
Step 8 adds `validate_and_map_citations` (selected-chunk ID membership only)
and `GroundedQuestionProcessor`: optional classify/filter → top-k store search
→ token-budget selection → evidence-bound `AnswerV1` → citation mapping onto
public `Citation` rows. OpenAI + a document store now builds the grounded
processor by default; the Step 5 tool loop stays available via direct
`LLMQuestionProcessor` construction. Corpus text is framed as untrusted
evidence; invented or budget-discarded citation IDs fail closed.

**Why it matters**
`POST /v1/questions` can return real citations tied to chunks actually
supplied to the model, and ID validation is explicitly not semantic
groundedness proof.

**Still fuzzy**
- Whether optional live classification should run on every question by default
- How much citation text to truncate for access-controlled deployments

**Next**
Step 9 bounded async I/O and concurrent ingestion.

See also `docs/decisions/grounded-answers.md`.

---

## 2026-11-25 — Chunking experiment

**What I figured out**
Step 7 added `ChunkingStrategy` / `ChunkingConfig` (recursive character,
token-aware, structure-aware), wired strategies through ingestion/seeding via
`CHUNKING_STRATEGY`, and shipped an isolated experiment harness plus
`python -m scripts.chunking_experiment`. The recorded fake-embedder run on
`data/raw` tied all three strategies at 30/30 answer-bearing hits, so production
keeps `recursive_character` until a larger corpus or live embeddings shows a
difference. Results live in `docs/evals/chunking-experiment-results.json` and
`docs/decisions/chunking.md`.

**Why it matters**
Chunking choices stay comparable without touching the live `documents` index,
and the decision is tied to actual metrics instead of intuition.

**Still fuzzy**
- Whether structure-aware wins once sources are long enough to split
- How MiniLM ranking would reorder the same 30 queries across strategies

**Next**
Step 8 retrieval-grounded answers.
(Completed later; see the 2026-11-26 entry.)

See also `docs/decisions/chunking.md`.

---

## 2026-11-24 — Embeddings and retrieval

**What I figured out**
Step 6 added an `EmbeddingClient` (deterministic fake by default, optional
local sentence-transformers), recursive character chunking with stable
`{source_id}:{ordinal}` IDs, an ingestion pipeline that delete-by-source then
upserts, an in-memory cosine vector `DocumentStore`, optional pgvector, and
retrieval-only `POST /v1/search`. Lifespan seeds `data/raw` and wires the same
vector store into OpenAI document tools. Thirty representative queries live
under `tests/evals/` with an automated source-id check.

**Why it matters**
Tools and HTTP search share real embedding retrieval behind the Step 5
protocol without adding generation to search. Re-index rules avoid silently
serving stale chunks when a source changes.

**Still fuzzy**
- Whether fake-embedder eval hits will transfer to MiniLM without retuning
  queries or chunk sizes
- When memory vs pgvector should become the Compose default for demos

**Next**
Step 7 chunking experiment; then Step 8 retrieval-grounded answers.
(Step 7 completed later; see the 2026-11-25 entry.)

See also `docs/decisions/embeddings-retrieval.md` and
`docs/evals/retrieval-inspection.md`.

---

## 2026-11-22 — Safe read-only tool calls before vector retrieval

**What I figured out**
Step 5 added a `DocumentStore` protocol with an `InMemoryDocumentStore`,
allow-listed `search_documents` / `get_document_chunk` tools, a
`SafeToolExecutor` (validate, deadline, size cap, safe error observations),
`LLMClient.generate_tool_turn`, and a bounded `ToolLoop`. The OpenAI question
path runs the loop when a store is injected; the local path stays tool-free.
Budgeting counts real tool schema strings. Public HTTP shape is unchanged;
citations stay empty; `RetrievalMetrics` can reflect tool activity.

**Why it matters**
The model can look up documents under hard allow-list and loop bounds without
shell/SQL/write tools, and without waiting for embeddings. Step 6 can replace
the in-memory store behind the same interface.

**Still fuzzy**
- How classification filters will steer which tools or metadata filters run
- Whether live OpenAI tool-calling latency will force a durable worker sooner

**Next**
Step 6 embeddings and retrieval; connect tools to a real vector-backed store.
(Completed later; see the 2026-11-24 entry.)

See also `docs/decisions/tools.md`.

---

## 2026-11-20 — Typed model outputs before tools and retrieval

**What I figured out**
Step 4 added versioned `ClassificationV1` and `AnswerV1` schemas under
`app/domain/model_outputs.py`, a `StructuredOutputError` for controlled
failures, and `LLMClient.generate_structured` behind the existing provider
boundary. OpenAI uses Responses `parse` with `text_format`, then re-validates
locally. The LLM question processor maps `AnswerV1.answer` onto the public
`QuestionResult` while citations stay empty. Incomplete structured finishes get
one retry; refusals and invariant violations fail closed. Classification is
tested but not wired into the live question path yet.

**Why it matters**
Schema-constrained output gives a stable place for claims, citation chunk IDs,
and abstention before tool loops or retrieval grounding. Versioning avoids
silently breaking callers when the contract changes.

**Still fuzzy**
- Whether public responses should later expose claims / insufficient_evidence
  under a versioned API change
- How classification filters will interact with real retrieval metadata

**Next**
Step 5 safe read-only tool calls (search/get chunk) with allow-listed tools.
(Completed later; see the 2026-11-22 entry.)

See also `docs/decisions/typed-outputs.md`.

---

## 2026-11-20 — Deterministic context budgeting before retrieval

**What I figured out**
Step 3 added `app/services/token_budget.py` with injectable token counters
(tiktoken for OpenAI-oriented models, character estimates for local) and
`allocate_context_budget`. Available retrieval capacity is the context window
minus instructions, question, history, tool schemas, framing, the output
reserve, and a safety margin. Ranked chunks are selected greedily; oversized
chunks are skipped. Fixed input that alone exceeds the window raises
`ContextBudgetExceededError` so the provider is never called with an oversized
request. Processors run the budget with an empty chunk list today and log only
safe counts.

**Why it matters**
Retrieval can later hand ranked chunks to the same allocator without changing
the public HTTP contract. Estimate labeling and the safety margin keep budget
math honest when exact provider accounting is unavailable.

**Still fuzzy**
- How large real tool schemas and multi-turn history will squeeze retrieval
- Whether citation text should be truncated separately from chunk selection

**Next**
Step 4 typed model outputs, still before full retrieval grounding.

---

## 2026-11-13 — Awaited generation and kept SQLite off the event loop

**What I figured out**
Step 2.5 made `LLMClient.generate` async with `AsyncOpenAI`, owned the shared
client in FastAPI lifespan, and awaited generation through the processor,
workflow, and routes. The SQLite store stayed synchronous; each workflow store
call runs in `asyncio.to_thread` and finishes before the provider await, so a
database lock is never held across generation. Injected processors use
`is not None` instead of truthiness. Cancelling a submit task during generate
leaves no terminal job and does not wedge the store for later requests.

**Why it matters**
The public question contract stays the same while the event loop can keep
serving health checks and other work during provider waits or blocked DB I/O.
Async alone is not a concurrency limit and is not a durable worker for jobs that
must survive disconnects.

**Still fuzzy**
- Whether real RAG latency will force a durable pending/completed worker
- How cancellation should interact with non-idempotent provider retries in
  production traffic

**Next**
Step 3 token and context budgeting before retrieval is wired into answers.
(Completed later; see the 2026-11-20 entry.)

---

## 2026-11-10 — Organized the API by responsibility

**What I figured out**
The FastAPI entry point is `app/main.py`; routers handle HTTP endpoints,
dependencies provide the shared workflow, and public schemas are separate
from service models. The lifespan creates and closes the SQLite store, which
is injected into the workflow. Submission runs synchronously and saves a
completed or failed job; an idempotency key can reuse a saved terminal result.

The local processor returns a placeholder, while the OpenAI adapter in
`app/llm/` can generate an answer when configured. Neither path performs
retrieval yet, and the optional pgvector service is not connected to the app.

**Why it matters**
Separating HTTP handling, orchestration, persistence, and generation makes each
boundary easier to test and evolve without changing the public API or existing
SQLite jobs.

**Still fuzzy**
- Which chunking and retrieval strategy works best for the intended documents
- Whether processing latency will eventually require a background worker

**Next**
Build an ingestion-to-retrieval vertical slice and evaluate retrieval and
answer quality before connecting retrieval to generation.

---

## 2026-10-23 — Built a runnable API and job-storage foundation

**What I figured out**
The API now has liveness and readiness routes, synchronous question submission,
and a status endpoint backed by SQLite. FastAPI's lifespan owns the shared
service lifecycle, and request middleware adds correlation IDs and request
metadata to logs. Docker has separate runtime and development targets, persistent
SQLite storage, and an optional pgvector service. The application still returns
a placeholder answer and does not connect to Postgres.

**Why it matters**
The API contract, persistence behavior, and failure responses can be exercised
without a real model or retrieval provider. The optional database container is
infrastructure only; it does not mean the RAG data path exists yet.

**Still fuzzy**
- Which chunking and retrieval strategy works best for the intended documents
- What evaluation set and retrieval/answer-quality measures to use
- Whether real processing latency will require a background worker

**Next**
Build a small ingestion-to-retrieval vertical slice and define a basic evaluation
set before adding real answer generation.

---

## 2026-10-22 — Make SQLite job updates explicit

**What I figured out**
The question service serializes access to its shared SQLite connection, uses a
timeout for temporary lock contention, and enables WAL for file-backed storage.
Creating and completing jobs use `INSERT` and `UPDATE`, so duplicate IDs and
missing jobs remain visible errors instead of being silently overwritten.

**Why it matters**
Persistence behavior should be deliberate: a retry or bad ID must not silently
replace a different job. SQLite is suitable for this single-process scaffold,
while deployment across multiple workers still needs a clear storage decision.

**Still fuzzy**
- How the job store should change if the API runs with multiple worker processes
- Whether question-job storage should remain SQLite when vector storage moves to Postgres

**Next**
Keep job-state storage and future vector storage as separate concerns until a
concrete deployment requirement calls for changing that boundary.

---

## 2026-10-21 — Treat request failures as part of the API contract

**What I figured out**
Liveness and readiness check different things: `/health` reports that the
process is running, while `/ready` checks SQLite. Validation failures return a
concise client error, unexpected failures return a generic server error, and a
request ID connects the response to request logs.

**Why it matters**
Failure paths need stable responses and useful diagnostics, without exposing
implementation details to API clients. Tests now cover malformed input,
service failures, and request-ID behavior.

**Still fuzzy**
- Which exception details from future retrieval and model providers are safe to log
- How to keep provider-specific errors useful while preventing sensitive data leaks

**Next**
Sanitize provider exceptions when external adapters are introduced.

---

## 2026-09-18 — Started a dedicated RAG lab

**What I figured out**
A real RAG system is a pipeline, not a single model call. Documents get ingested, chunked, embedded, stored, retrieved, ranked, and then passed to a generator. This repo is the place to learn those pieces by building and evaluating them.

**Why it matters**
Without a lab and a written trail, it is easy to try tools without remembering *why* a choice worked or failed. Logging the “aha” moments keeps the architecture decisions honest.

**Still fuzzy**
- How chunk size and overlap actually affect retrieval quality
- When keyword search beats (or should mix with) vector search
- What a useful eval set looks like for this project

**Next at the time**
Stand up Postgres with pgvector and build the first ingestion → embedding →
retrieval loop in the application.
