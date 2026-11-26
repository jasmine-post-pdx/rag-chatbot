# Decision: Safe read-only document tools (Step 5)

## Context

Before vector retrieval (Step 6) and citation grounding (Step 8), the model
needs a constrained way to look up documents. Unrestricted tools (shell, SQL,
arbitrary file writes) are out of scope for this lab and unsafe for untrusted
model arguments.

## Decision

Expose exactly two allow-listed tools:

- `search_documents(query, framework=None, language=None, limit=5)`
- `get_document_chunk(chunk_id)`

Both are read-only. They call a `DocumentStore` protocol. Step 5 ships an
`InMemoryDocumentStore` with a small non-secret sample corpus. Step 6 replaces
that adapter with a real retrieval backend behind the same interface.

Every proposed call is:

1. Checked against the allow-list (unknown names never execute).
2. Validated locally with Pydantic argument models (including a hard max on
   `limit`).
3. Executed under a per-call deadline (`TOOL_CALL_TIMEOUT_SECONDS`).
4. Size-capped before the observation returns to the model
   (`TOOL_MAX_RESULT_CHARS`).
5. Logged with request ID, tool name, sanitized argument metadata, duration,
   and success/failure — never full document text or prompt content.

A `ToolLoop` bounds model↔tool iterations (`TOOL_MAX_TURNS`) and total
executions (`TOOL_MAX_CALLS`). Invalid calls and handler failures become safe
error observations so the model can continue or abstain; runaway loops fail
closed as `StructuredOutputError` with category `unsupported`.

The tool loop remains available when `LLMQuestionProcessor` is constructed with
a `DocumentStore` directly. The factory default for OpenAI + store is now
Step 8 `GroundedQuestionProcessor` (retrieve → budget → citation-validated
answer) rather than the tool loop. The local placeholder path stays tool-free.
`RetrievalMetrics` may still reflect tool activity on the explicit tool path.

## Alternatives considered

- Provider-native retrieval only: deferred until Step 6 embeddings exist.
- Writing tools into the OpenAI SDK client: rejected so validation, bounds, and
  logging stay application-owned and unit-testable with fakes.
- Changing the public API to expose tool traces: deferred; keep the versioned
  question shape stable.

## Consequences

- Unit tests cover the store, executor, LLM tool-turn mapping, and full fake
  model → tool → observation → answer loop without live credentials.
- Step 6 can swap `InMemoryDocumentStore` without rewriting tools or the loop.
- Step 8 validates citation IDs against chunks selected for the prompt (see
  [`grounded-answers.md`](grounded-answers.md)); allow-listed tool lookup alone
  does not prove groundedness.
