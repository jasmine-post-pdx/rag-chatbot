# Decision: Retrieval-grounded answers (Step 8)

## Context

After embeddings, search, budgeting, typed outputs, and document tools, the
question path still returned empty `citations`. Step 8 integrates retrieval
into `POST /v1/questions` with evidence-bound generation and citation checks.

## Decision

- **`GroundedQuestionProcessor`** runs: optional classifier filters →
  `DocumentStore.search_documents` (top-k, off the event loop) →
  `ContextBudgetConfig.allocate` → `generate_structured(AnswerV1)` on the
  budget-assembled evidence prompt → `validate_and_map_citations`.
- **Citation IDs** must belong to chunks **selected for the prompt**, not merely
  retrieved. Invented or budget-discarded IDs raise
  `StructuredOutputError(category="validation")`.
- **ID membership ≠ semantic groundedness.** Validation does not check that
  claim text is entailed by the cited chunk.
- **Corpus text is untrusted reference** under the budget `Evidence:` framing;
  application instructions stay separate.
- **Factory wiring:** `build_question_processor("openai", ..., document_store=store)`
  returns `GroundedQuestionProcessor`. The Step 5 tool-loop
  `LLMQuestionProcessor` remains constructible explicitly for tool tests.
- Logs omit prompt, chunk, and answer bodies; retrieval metrics expose
  `documents_considered` (hits) and `chunks_returned` (selected).

## Alternatives considered

- Keep the tool loop as the default OpenAI+store path: rejected for Step 8’s
  explicit retrieve → budget → generate contract.
- Validate citations against all retrieved hits: rejected; only selected
  evidence was supplied to the model.
- Expose claims / `insufficient_evidence` on the public HTTP schema: deferred;
  the existing `citations` list is enough for this step.

## Consequences

- Public responses can return non-empty `citations` with `source_id`,
  `chunk_id`, and excerpt `text`.
- Unit/integration tests cover in-corpus answers, abstention, invented IDs,
  over-budget selection, prompt-injection corpus text, provider failure, and
  retrieval improvement with fakes only.
