# Learning Progress

This file is a running log of what I figure out while learning AI tech — especially retrieval-augmented generation, embeddings, evaluation, and the systems around them.

The point is not a polished essay. Each entry should capture what clicked, why it matters, and what is still fuzzy so I can see the trail later.

Newest entries go at the top.

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
