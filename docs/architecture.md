# Architecture

This lab is organized around a retrieval-augmented generation (RAG) pipeline: ingest documents, retrieve relevant context, then generate an answer.

```text
client
  -> app/api
    -> app/services
      -> app/retrieval
      -> app/models
      -> app/tools
```

## Application

| Path | Responsibility |
| --- | --- |
| `app/api/` | HTTP and other entrypoints. Keep handlers thin: parse input, call a service, return a response. |
| `app/core/` | Cross-cutting concerns such as centralized configuration. Keep this layer free of business logic. |
| `app/services/` | Orchestration. Compose retrieval, model calls, and tools into a request flow. |
| `app/models/` | LLM and embedding adapters. Isolate provider-specific code here. |
| `app/retrieval/` | Chunking, indexing, search, ranking, and context assembly. |
| `app/tools/` | Optional callable tools used during generation. |

## Data

| Path | Responsibility |
| --- | --- |
| `data/raw/` | Source documents as ingested, unmodified. |
| `data/processed/` | Chunks, embeddings, and other derived artifacts. |

Processing scripts live in `scripts/` and should write from `data/raw/` into `data/processed/` rather than mutating sources in place.

## Tests

| Path | Responsibility |
| --- | --- |
| `tests/unit/` | Fast, isolated checks of retrieval, ranking, and service logic. |
| `tests/integration/` | Checks that exercise storage, APIs, or composed services. |
| `tests/evals/` | Quality evaluations: retrieval hit rate, groundedness, answer faithfulness. |

## Decisions

Record architectural choices in `docs/decisions/` so later experiments can compare against a known baseline rather than reconstructing why a chunker, index, or model was selected.
