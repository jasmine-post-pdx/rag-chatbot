# Other API sequence diagrams

These are the health routes in `app/routers/health.py`: `/health` and
`/ready`. Both requests pass through the request logging middleware.

## Text preview

```text
Client          Middleware       FastAPI route       Workflow       SQLite store       SQLite database
  |                |                 |                 |                |                |
  | GET /health    |                 |                 |                |                |
  |--------------->|                 |                 |                |                |
  |                | add request ID |                 |                |                |
  |                |---------------->|                 |                |                |
  |                |                 | return {ok}     |                |                |
  |                |<----------------| 200 response    |                |                |
  |<---------------| response + x-request-id           |                |                |
  |                |                 |                 |                |                |
  | GET /ready     |                 |                 |                |                |
  |--------------->|---------------->|                 |                |                |
  |                |                 | is_ready()      |                |                |
  |                |                 |---------------->|                |                |
  |                |                 |                 | is_ready()     |                |
  |                |                 |                 |--------------->|                |
  |                |                 |                 |                | SELECT request_id FROM question_jobs LIMIT 1
  |                |                 |                 |                |--------------->|
  |                |                 |                 |                |<---------------|
  |                |                 |<----------------| True/False      |                |
  |                |                 |                 |                |                |
  |<---------------|<----------------| 200 ready or 503 unavailable      |                |
```

```mermaid
sequenceDiagram
    actor Client
    participant MW as RequestLogMiddleware
    participant API as FastAPI route
    participant WF as QuestionWorkflow
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
    MW->>API: Forward request
    API->>WF: is_ready()
    WF->>Store: is_ready()
    Store->>DB: SELECT request_id FROM question_jobs LIMIT 1
    DB-->>Store: Row or empty result

    alt SQLite query succeeds
        Store-->>WF: True
        WF-->>API: True
        API-->>MW: 200 {status: ready}
        MW-->>Client: 200 + x-request-id
    else SQLite query raises an error
        Store-->>WF: False
        WF-->>API: False
        API-->>MW: 503 Service unavailable
        MW-->>Client: 503 + x-request-id
    end
```
