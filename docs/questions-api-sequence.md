# Question API sequence diagrams

The routes in `app/routers/questions.py` use the shared `QuestionWorkflow`
from `app/dependencies/questions.py`. `app/middleware.py` supplies the HTTP
correlation ID; the workflow generates a separate question job ID. Public
request and response models live in `app/schemas/`, separate from service DTOs.

## Submit a question

`POST /v1/questions` processes the question synchronously and returns the
completed result. The request body contains `question`; whitespace is stripped
and the resulting value must contain 1–4,000 characters. An optional
`Idempotency-Key` must contain 1–255 characters and cannot be blank.

```mermaid
sequenceDiagram
    actor Client
    participant MW as Request middleware
    participant API as Question router
    participant Err as Application error handler
    participant Dep as Workflow dependency
    participant WF as QuestionWorkflow
    participant Store as SQLiteQuestionJobStore
    participant Processor as Question processor
    participant LLM as OpenAI adapter

    Client->>MW: POST /v1/questions (question, optional Idempotency-Key)
    MW->>MW: Create/normalize x-request-id
    MW->>API: Forward request
    API->>Dep: get_question_workflow(request)
    Dep-->>API: Shared workflow from app state
    API->>API: Validate question and idempotency key
    API->>WF: submit_question(question, correlation ID, key)
    WF->>WF: Generate job ID
    opt Idempotency key supplied
        WF->>Store: Look up hashed key
        Store-->>WF: Existing job and fingerprint, or none
    end
    alt Matching completed job exists
        WF-->>API: Saved result
    else Key belongs to a different question
        WF-->>API: IdempotencyKeyReuseError
        API-->>MW: 409 Conflict
        MW-->>Client: 409 with x-request-id
    else Prior job failed or is incomplete
        WF-->>API: QuestionProcessingError
        API-->>MW: Propagate error
        MW-->>Err: Propagate error
        Err-->>Client: Generic 500 with x-request-id and x-question-id
    else No saved job for the key
        WF->>Processor: process(job ID, question, correlation ID)
        opt OpenAI provider configured
            Processor->>LLM: generate(question)
            LLM-->>Processor: Answer and model usage, or error
        end
        alt Processing succeeds
            Processor-->>WF: Result DTO
            WF->>Store: Insert completed job
            Store-->>WF: Persisted terminal job
            WF-->>API: Result DTO
        else Processing fails
            Processor-->>WF: Processing error
            WF->>Store: Insert failed job (processing_error)
            Store-->>WF: Persisted terminal job
            WF-->>API: QuestionProcessingError
            API-->>MW: Propagate error
            MW-->>Err: Propagate error
            Err-->>Client: Generic 500 with x-request-id and x-question-id
        end
    end
    opt A result was returned
        API->>API: Map DTO to public response schema
        API-->>MW: 200 QuestionResult
        MW-->>Client: 200 with x-request-id
    end
```

The configured local processor returns a placeholder; the OpenAI processor
uses the Responses API. Neither performs retrieval: citations are empty and
retrieval counts are zero. No pending state or in-flight idempotency claim is
stored by this workflow. Simultaneous requests with the same new key may both
process; the first saved terminal job wins, and a different question for that
key produces a conflict. A saved failed or incomplete job returns a generic
500 on retry rather than being processed again. The diagram shows the usual
non-racing path; in a race, insertion can return the other request's job or a
key-conflict error instead.

Invalid request data or keys return 422. Other unexpected errors, including
failure to persist a failed job or an invalid server result, return a generic
500; `x-question-id` is included for `QuestionProcessingError` only. The
`x-request-id` response header is the HTTP correlation ID, not the job ID.

## Retrieve a question job

```mermaid
sequenceDiagram
    actor Client
    participant MW as Request middleware
    participant API as Question router
    participant Dep as Workflow dependency
    participant WF as QuestionWorkflow
    participant Store as SQLiteQuestionJobStore

    Client->>MW: GET /v1/questions/{request_id}
    MW->>MW: Create/normalize x-request-id
    MW->>API: Forward request
    API->>Dep: get_question_workflow(request)
    Dep-->>API: Shared workflow from app state
    API->>WF: get_job(request_id)
    WF->>Store: get_job(request_id)
    Store-->>WF: Stored job or none
    WF-->>API: Stored job or none
    alt Job exists
        API->>API: Map job and any result to public schemas
        API-->>MW: 200 QuestionJobResponse
    else Job not found
        API-->>MW: 404 Question not found
    end
    MW-->>Client: Response with x-request-id
```

The job response contains `request_id`, `status`, `result`, and
`error_category`. A completed job has a result; a failed job has the safe
`processing_error` category. Legacy pending jobs can still be read, although
the current submission workflow persists only terminal jobs. Archived jobs are
not returned by this endpoint.