# Bounded concurrent ingestion

## Decision

Use `BoundedIngestionPipeline` for async, concurrency-capped document ingestion.
Cap concurrent embedding calls with `INGESTION_CONCURRENCY` (default 5, max 10).
Offload blocking parse, chunking, and sync vector-store drivers with
`asyncio.to_thread`. Record per-source failures without aborting siblings.
Honor per-source and overall deadlines; cancel drains in-flight work.

Keep ingest as a simple CLI (`python -m scripts.ingest`) plus lifespan seed.
Do not add a durable ingest queue unless a real requirement appears.

## Why

Step 9 calls for genuine async I/O where supported and bounded concurrent
ingest, not unbounded `gather` of every source. Provider rate limits and
local CPU encode both benefit from a hard concurrency ceiling. Question
serving stays isolated because the CLI is a separate process; lifespan seed
awaits the same bounded path and does not hold the event loop on sync I/O.

## Behavior

| Concern | Policy |
| --- | --- |
| Concurrency unit | Concurrent `aembed_documents` calls under a semaphore |
| Store mutations | Serialized with an asyncio lock (safe for in-memory store) |
| Re-index | Per-source delete-before-upsert (unchanged from Step 6) |
| Partial failure | Continue siblings; `category` is `embed`, `upsert`, `timeout`, or `error` |
| Overall timeout | Soft-fail: cancel pending sources, return timeout failures |
| Cancel | Propagate `CancelledError`; leave pipeline reusable |
| Logging | Counts and categories only; no document bodies |

## How to run

```bash
# Separate process from the API — does not share the API event loop
INGESTION_CONCURRENCY=5 python -m scripts.ingest --path data/raw
```

Lifespan seeding of `CORPUS_RAW_DIR` uses the same settings when the API starts.

## Tradeoffs

- Sync pgvector/memory stores stay sync; offload is enough for this lab.
- Soft overall timeout prefers CLI exit 0 with failure counts over a hard raise.
- A durable queue would help multi-host or crash-resume jobs; not needed yet.
