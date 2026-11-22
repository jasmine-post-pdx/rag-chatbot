"""Versioned schemas for schema-constrained model outputs.

These internal DTOs describe structured classification and answer payloads
returned by a language model. They are not part of the public HTTP contract.
Confidence values are model claims, not calibrated probabilities.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassificationV1(BaseModel):
    """Narrow request classification before retrieval or tool use.

    :ivar schema_version: Contract version; bump deliberately rather than
        silently changing field meaning.
    :ivar intent: High-level handling intent claimed by the model.
    :ivar needs_retrieval: Whether the model claims document retrieval is needed.
    :ivar framework: Optional framework filter for retrieval, or ``None``.
    :ivar language: Optional language filter for retrieval, or ``None``.
    :ivar confidence: Model-claimed confidence in ``[0, 1]``, not calibrated.
    :ivar clarifying_question: Follow-up question when intent is clarify, else
        ``None``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    intent: Literal["answer", "clarify", "out_of_scope"]
    needs_retrieval: bool
    framework: str | None = None
    language: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    clarifying_question: str | None = None


class AnswerClaim(BaseModel):
    """A material claim and the chunk IDs offered as support.

    :ivar claim: Claim text the answer relies on.
    :ivar citation_chunk_ids: Chunk identifiers cited for this claim."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    citation_chunk_ids: list[str]


class AnswerV1(BaseModel):
    """Structured answer with evidence claims and an abstention flag.

    When ``insufficient_evidence`` is ``True``, ``claims`` must be empty.
    When it is ``False``, every claim must include at least one citation chunk
    ID. Citation IDs are not verified against a retrieval set here; that check
    belongs to later retrieval-grounded steps.

    :ivar schema_version: Contract version; bump deliberately rather than
        silently changing field meaning.
    :ivar answer: Answer text shown to the caller.
    :ivar claims: Material claims with citation chunk IDs.
    :ivar insufficient_evidence: Whether the model abstains for lack of evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    answer: str
    claims: list[AnswerClaim]
    insufficient_evidence: bool

    @model_validator(mode="after")
    def _enforce_citation_invariants(self) -> "AnswerV1":
        """Reject inconsistent claim and insufficient-evidence combinations.

        :returns: This instance when invariants hold.
        :raises ValueError: If claims are present with insufficient evidence, or
            a supported claim is missing citation chunk IDs."""
        if self.insufficient_evidence:
            if self.claims:
                raise ValueError(
                    "claims must be empty when insufficient_evidence is true"
                )
            return self
        for index, claim in enumerate(self.claims):
            if not claim.citation_chunk_ids:
                raise ValueError(
                    f"claims[{index}] must include at least one citation_chunk_id "
                    "when insufficient_evidence is false"
                )
        return self
