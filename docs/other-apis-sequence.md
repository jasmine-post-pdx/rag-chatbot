# Other API sequence diagrams

These are the health routes in `app/routers/health.py`: `/health` and
`/ready`. Both requests pass through `app/middleware.py`. `/ready` obtains the
shared question-job store through `app/dependencies/question_jobs.py`.

## Text preview

```text
Client          Middleware       FastAPI route       SQLite store       SQLite database
  |                |                 |                 |                |
  | GET /health    |                 |                 |                |
  |--------------->|                 |                 |                |
  |                | add request ID |                 |                |
  |                |---------------->|                 |                |
  |                |                 | {"status":"ok"} |                |
  |                |<----------------| 200 response    |                |
  |<---------------| response + x-request-id           |                |
  |                |                 |                 |                |
  | GET /ready     |                 |                 |                |
  |--------------->|---------------->|                 |                |
  |                |                 | await to_thread(is_ready) |      |
  |                |                 |---------------->|                |
  |                |                 |                 | SELECT question_id FROM question_jobs LIMIT 1
  |                |                 |                 |--------------->|
  |                |                 |                 |<---------------|
  |                |                 |<----------------| True/False      |
  |                |                 |                 |                |
  |<---------------|<----------------| 200 ready or 503 unavailable      |
```

```mermaid
sequenceDiagram
    actor Client
    participant MW as RequestLogMiddleware
    participant API as Health router
    participant Dep as Job store dependency
    participant Store as SQLiteQuestionJobStore
    participant DB as SQLite database

    Note over Client,DB: GET /health
    Client->>MW: GET /health
    MW->>MW: Create/normalize x-request-id
    MW->>API: Forward request
    API-->>MW: 200 {status: ok}
    MW-->>Client: 200 + x-request-id

    Note over Client,DB: GET /ready
    Client->>MW: GET /ready
    MW->>MW: Create/normalize x-request-id
    MW->>API: Forward request
    API->>Dep: get_question_job_store(request)
    Dep-->>API: Shared store from app state
    API->>Store: await to_thread(is_ready)
    Store->>DB: SELECT question_id FROM question_jobs LIMIT 1
    DB-->>Store: Row or empty result

    alt SQLite query succeeds
        Store-->>API: True
        API-->>MW: 200 {status: ready}
        MW-->>Client: 200 + x-request-id
    else SQLite query raises an error
        Store-->>API: False
        API-->>MW: 503 Service unavailable
        MW-->>Client: 503 + x-request-id
    end
```

## POST /v1/search

Retrieval-only search in `app/routers/search.py`. The route obtains
`SearchService` through `app/dependencies/search.py` and offloads the store
lookup with `asyncio.to_thread`.

```text
Client          Middleware       Search route       SearchService       Vector store
  |                |                 |                 |                |
  | POST /v1/search|                 |                 |                |
  |--------------->|                 |                 |                |
  |                | add request ID |                 |                |
  |                |---------------->|                 |                |
  |                |                 | await to_thread(search) |        |
  |                |                 |---------------->|                |
  |                |                 |                 | embed query + top-k
  |                |                 |                 |--------------->|
  |                |                 |                 |<---------------|
  |                |                 |<----------------| SearchResponse |
  |                |<----------------| 200 hits        |                |
  |<---------------| response + x-request-id           |                |
```

```mermaid
sequenceDiagram
    actor Client
    participant MW as RequestLogMiddleware
    participant API as Search router
    participant Dep as SearchService dependency
    participant Svc as SearchService
    participant Store as VectorDocumentStore

    Client->>MW: POST /v1/search
    MW->>MW: Create/normalize x-request-id
    MW->>API: Forward request
    API->>Dep: get_search_service(request)
    Dep-->>API: Shared service from app state
    API->>Svc: await to_thread(search)
    Svc->>Store: search_documents(query, filters, top_k)
    Store-->>Svc: ranked hits
    Svc-->>API: SearchResponse
    API-->>MW: 200 hits + embedding_model + similarity_metric
    MW-->>Client: 200 + x-request-id
```
