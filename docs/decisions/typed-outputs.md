# Typed model outputs (schema version 1)

## Decision

Model outputs used for generation are versioned Pydantic schemas in
`app/domain/model_outputs.py`, not public HTTP models. The OpenAI question path
requests `AnswerV1` through `LLMClient.generate_structured` (Responses
`parse` + local re-validation). `ClassificationV1` is defined and tested but
not called from `POST /v1/questions` yet.

## Contracts

- `schema_version` is `Literal["1"]`. Bumping the contract means a new Literal
  or model, not silent field renames.
- Extra fields are forbidden (`extra="forbid"`).
- Classification `confidence` is a model claim in `[0, 1]`, not a calibrated
  probability.
- Answer invariants: `insufficient_evidence=True` requires empty `claims`;
  otherwise every claim needs at least one `citation_chunk_id`.

## Recovery

- Refusal → `StructuredOutputError(category="refusal")` (no invented answer).
- Incomplete finish → one bounded re-request, then the same controlled failure.
- Local validation / missing parsed payload → `validation` (or `unsupported`
  when the provider lacks schema-constrained parse).
- Public HTTP responses still expose `answer: str` and empty `citations` until
  retrieval grounding maps claims to citation excerpts.
