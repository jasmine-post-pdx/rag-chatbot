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
  |                |                 | return {ok}     |                |
  |                |<----------------| 200 response    |                |
  |<---------------| response + x-request-id           |                |
  |                |                 |                 |                |
  | GET /ready     |                 |                 |                |
  |--------------->|---------------->|                 |                |
  |                |                 | is_ready()      |                |
  |                |                 |---------------->|                |
  |                |                 |                 | SELECT request_id FROM question_jobs LIMIT 1
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
    API->>Store: is_ready()
    Store->>DB: SELECT request_id FROM question_jobs LIMIT 1
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
