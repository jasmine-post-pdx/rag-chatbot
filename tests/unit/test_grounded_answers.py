"""Unit tests for Step 8 retrieval-grounded question answering.

These tests drive ``GroundedQuestionProcessor`` with fakes only. They cover the
acceptance cases: in-corpus citations, out-of-corpus abstention, invented
citations, over-budget selection, provider failure, retrieval trace, retrieval
improvement, evidence framing, logging hygiene, and optional classify/filter.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from app.domain.errors import StructuredOutputError
from app.domain.model_outputs import AnswerClaim, AnswerV1, ClassificationV1
from app.domain.questions import Citation
from app.llm import FakeLLMClient, StructuredGenerationResult
from app.retrieval.document_store import DocumentChunk
from app.services.question_processor import GroundedQuestionProcessor
from app.services.token_budget import ApproximateTokenCounter, ContextBudgetConfig
from tests.support.overlap_store import TokenOverlapDocumentStore


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "fastapi-guide",
    framework: str | None = "fastapi",
    language: str | None = "python",
) -> DocumentChunk:
    """Build a document chunk for grounded processor fixtures.

    :param chunk_id: Stable chunk identifier.
    :param text: Chunk body text.
    :param source_id: Source document identifier.
    :param framework: Optional framework filter metadata.
    :param language: Optional language filter metadata.
    :returns: A ``DocumentChunk`` instance."""
    return DocumentChunk(
        source_id=source_id,
        chunk_id=chunk_id,
        text=text,
        framework=framework,
        language=language,
    )


def _corpus_store(*extra: DocumentChunk) -> TokenOverlapDocumentStore:
    """Return a store with FastAPI Depends evidence plus optional extras.

    :param extra: Additional chunks to index.
    :returns: A ``TokenOverlapDocumentStore`` with a generous search max."""
    chunks = [
        _chunk(
            "fastapi-deps",
            "FastAPI dependency injection wires routers to services. "
            "Use Depends to inject shared resources into path operations.",
        ),
        _chunk(
            "fastapi-lifespan",
            "FastAPI lifespan handlers create shared clients on startup "
            "and close them on shutdown.",
        ),
        *extra,
    ]
    return TokenOverlapDocumentStore(chunks, max_limit=20)


def _answer(
    text: str,
    *,
    insufficient_evidence: bool = False,
    claims: list[AnswerClaim] | None = None,
) -> AnswerV1:
    """Build a structured answer for fake clients.

    :param text: Answer text.
    :param insufficient_evidence: Abstention flag.
    :param claims: Optional claims; empty when abstaining.
    :returns: A validated ``AnswerV1``."""
    if insufficient_evidence:
        return AnswerV1(answer=text, claims=[], insufficient_evidence=True)
    return AnswerV1(
        answer=text,
        claims=claims
        if claims is not None
        else [
            AnswerClaim(
                claim="Depends injects shared resources",
                citation_chunk_ids=["fastapi-deps"],
            )
        ],
        insufficient_evidence=False,
    )


def _budget(
    *,
    context_window_tokens: int = 128000,
    output_reserve_tokens: int = 64,
    safety_margin_tokens: int = 0,
    instructions: str | None = None,
) -> ContextBudgetConfig:
    """Build a context budget with the approximate counter.

    :param context_window_tokens: Model context-window size.
    :param output_reserve_tokens: Tokens reserved for model output.
    :param safety_margin_tokens: Extra estimate margin.
    :param instructions: Optional instructions counted as fixed input.
    :returns: A ``ContextBudgetConfig`` instance."""
    return ContextBudgetConfig(
        counter=ApproximateTokenCounter(),
        context_window_tokens=context_window_tokens,
        output_reserve_tokens=output_reserve_tokens,
        safety_margin_tokens=safety_margin_tokens,
        instructions=instructions,
    )


class RecordingStructuredClient:
    """Record prompts passed to ``generate_structured`` and return a fixed answer.

    :ivar prompts: Prompts received, in call order.
    :ivar _parsed: Structured answer returned on each call."""

    def __init__(self, parsed: AnswerV1) -> None:
        """Initialize with a canned structured answer.

        :param parsed: Answer returned for every structured generation call."""
        self.prompts: list[str] = []
        self._parsed = parsed
        self.provider = "fake"
        self.model = "fake-grounded"

    async def generate_structured(self, prompt: str, output_model: type):
        """Record the prompt and return the canned structured result.

        :param prompt: Prompt text to record.
        :param output_model: Expected output model type.
        :returns: Structured result wrapping the canned answer."""
        assert output_model is AnswerV1
        self.prompts.append(prompt)
        return StructuredGenerationResult(
            parsed=self._parsed,
            provider=self.provider,
            model=self.model,
            input_tokens=11,
            output_tokens=22,
            finish_status="completed",
            provider_request_id="resp_grounded",
        )

    async def generate(self, question: str):
        """Satisfy the LLMClient protocol without being used by grounded path."""
        raise AssertionError("grounded path must use generate_structured")

    async def generate_tool_turn(self, *args: Any, **kwargs: Any):
        """Grounded path must not open a tool loop."""
        raise AssertionError("grounded path must not call generate_tool_turn")


class PromptAwareClient:
    """Return an abstention or cited answer depending on evidence in the prompt.

    Used to show that better retrieval can change the structured outcome when
    the fake inspects the assembled evidence text.

    :ivar prompts: Prompts received, in call order."""

    def __init__(self) -> None:
        """Initialize with an empty prompt log."""
        self.prompts: list[str] = []
        self.provider = "fake"
        self.model = "prompt-aware"

    async def generate_structured(self, prompt: str, output_model: type):
        """Abstain unless Depends evidence appears under the Evidence framing.

        The user question may itself contain the word Depends; only corpus
        evidence above the Question section counts as retrieval improvement.

        :param prompt: Assembled evidence-bound prompt.
        :param output_model: Expected output model type.
        :returns: Cited answer when Depends evidence is present, else abstention."""
        assert output_model is AnswerV1
        self.prompts.append(prompt)
        evidence_section = prompt.split("Question:", 1)[0] if "Question:" in prompt else ""
        if "Evidence:" in evidence_section and "Depends" in evidence_section:
            parsed = _answer(
                "FastAPI uses Depends for dependency injection.",
                claims=[
                    AnswerClaim(
                        claim="Depends injects shared resources",
                        citation_chunk_ids=["fastapi-deps"],
                    )
                ],
            )
        else:
            parsed = _answer(
                "I do not have enough evidence to answer.",
                insufficient_evidence=True,
            )
        return StructuredGenerationResult(
            parsed=parsed,
            provider=self.provider,
            model=self.model,
            input_tokens=5,
            output_tokens=7,
            finish_status="completed",
        )

    async def generate(self, question: str):
        """Satisfy the LLMClient protocol without being used."""
        raise AssertionError("grounded path must use generate_structured")

    async def generate_tool_turn(self, *args: Any, **kwargs: Any):
        """Grounded path must not open a tool loop."""
        raise AssertionError("grounded path must not call generate_tool_turn")


class FailingClient:
    """Simulate a provider that fails every structured generation request."""

    provider = "fake"
    model = "failing"

    async def generate_structured(self, prompt: str, output_model: type):
        """Raise a provider failure.

        :param prompt: Prompt that would have been answered.
        :param output_model: Expected output model type.
        :raises RuntimeError: Always, to simulate an unavailable provider."""
        raise RuntimeError("provider unavailable")

    async def generate(self, question: str):
        """Satisfy the LLMClient protocol without being used."""
        raise AssertionError("unused")

    async def generate_tool_turn(self, *args: Any, **kwargs: Any):
        """Grounded path must not open a tool loop."""
        raise AssertionError("unused")


@pytest.mark.asyncio
async def test_in_corpus_answer_returns_citations_and_retrieval_trace():
    """Answer an in-corpus question with citations and a visible retrieval trace."""
    client = RecordingStructuredClient(
        _answer(
            "FastAPI uses Depends for dependency injection.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["fastapi-deps"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        _corpus_store(),
        budget=_budget(),
        retrieval_top_k=5,
    )

    result = await processor.process(
        "req-in-corpus",
        "How does FastAPI inject dependencies with Depends?",
        request_id="http-in",
    )

    assert result.answer == "FastAPI uses Depends for dependency injection."
    assert result.citations == [
        Citation(
            source_id="fastapi-guide",
            chunk_id="fastapi-deps",
            text=(
                "FastAPI dependency injection wires routers to services. "
                "Use Depends to inject shared resources into path operations."
            ),
        )
    ]
    assert result.retrieval.documents_considered >= 1
    assert result.retrieval.chunks_returned >= 1
    assert result.model.provider == "fake"
    assert result.model.name == "fake-grounded"
    assert result.model.input_tokens == 11
    assert result.model.output_tokens == 22


@pytest.mark.asyncio
async def test_out_of_corpus_abstention_returns_empty_citations():
    """Abstain with empty citations when the store has no relevant evidence."""
    empty = TokenOverlapDocumentStore([], max_limit=5)
    client = RecordingStructuredClient(
        _answer(
            "I do not have enough evidence to answer.",
            insufficient_evidence=True,
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        empty,
        budget=_budget(),
        retrieval_top_k=5,
    )

    result = await processor.process(
        "req-ooc",
        "What is the capital of Atlantis?",
        request_id="http-ooc",
    )

    assert result.answer == "I do not have enough evidence to answer."
    assert result.citations == []
    assert result.retrieval.chunks_returned == 0


@pytest.mark.asyncio
async def test_invented_citations_fail_closed():
    """Reject model answers that cite chunk IDs never selected for the prompt."""
    client = RecordingStructuredClient(
        _answer(
            "FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["invented-chunk-id"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        _corpus_store(),
        budget=_budget(),
        retrieval_top_k=5,
    )

    with pytest.raises(StructuredOutputError) as raised:
        await processor.process(
            "req-invented",
            "How does FastAPI inject dependencies with Depends?",
            request_id="http-invented",
        )

    assert raised.value.category == "validation"


@pytest.mark.asyncio
async def test_over_budget_selection_excludes_discarded_chunks_from_prompt_and_citations():
    """Only budget-fitting chunks appear in the prompt; discarded IDs are not citeable."""
    # Include a Depends token so overlap search still retrieves the huge chunk.
    huge = "Depends " + ("X" * 4000)
    store = TokenOverlapDocumentStore(
        [
            _chunk("huge-chunk", huge, source_id="huge-doc"),
            _chunk(
                "small-chunk",
                "Depends injects shared resources into path operations.",
                source_id="fastapi-guide",
            ),
        ],
        max_limit=10,
    )
    # Tiny window: framing + huge text cannot fit; the small chunk can.
    budget = ContextBudgetConfig(
        counter=ApproximateTokenCounter(),
        context_window_tokens=200,
        output_reserve_tokens=20,
        safety_margin_tokens=0,
    )
    client = RecordingStructuredClient(
        _answer(
            "Depends injects shared resources.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["small-chunk"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=budget,
        retrieval_top_k=5,
    )

    result = await processor.process(
        "req-budget",
        "Depends",
        request_id="http-budget",
    )

    assert len(client.prompts) == 1
    prompt = client.prompts[0]
    assert "Depends injects shared resources" in prompt
    assert huge not in prompt
    assert all(citation.chunk_id != "huge-chunk" for citation in result.citations)
    assert any(citation.chunk_id == "small-chunk" for citation in result.citations)


@pytest.mark.asyncio
async def test_over_budget_discarded_chunk_id_cannot_be_cited():
    """Citing a retrieved-but-discarded chunk ID fails closed."""
    huge = "Depends " + ("Y" * 4000)
    store = TokenOverlapDocumentStore(
        [
            _chunk("huge-chunk", huge, source_id="huge-doc"),
            _chunk(
                "small-chunk",
                "Depends injects shared resources into path operations.",
            ),
        ],
        max_limit=10,
    )
    budget = ContextBudgetConfig(
        counter=ApproximateTokenCounter(),
        context_window_tokens=200,
        output_reserve_tokens=20,
        safety_margin_tokens=0,
    )
    client = RecordingStructuredClient(
        _answer(
            "Answer citing the discarded huge chunk.",
            claims=[
                AnswerClaim(
                    claim="Huge evidence supports this",
                    citation_chunk_ids=["huge-chunk"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=budget,
        retrieval_top_k=5,
    )

    with pytest.raises(StructuredOutputError) as raised:
        await processor.process("req-discard", "Depends", request_id="http-discard")

    assert raised.value.category == "validation"


@pytest.mark.asyncio
async def test_provider_failure_propagates_without_fabricated_citations():
    """Provider failures propagate; the processor does not invent an answer."""
    processor = GroundedQuestionProcessor(
        FailingClient(),
        _corpus_store(),
        budget=_budget(),
        retrieval_top_k=5,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await processor.process(
            "req-fail",
            "How does FastAPI inject dependencies with Depends?",
            request_id="http-fail",
        )


@pytest.mark.asyncio
async def test_retrieval_trace_reflects_retrieve_and_select():
    """Retrieval metrics are non-zero when evidence is retrieved and selected."""
    client = RecordingStructuredClient(_answer("FastAPI uses Depends."))
    processor = GroundedQuestionProcessor(
        client,
        _corpus_store(),
        budget=_budget(),
        retrieval_top_k=5,
    )

    result = await processor.process(
        "req-trace",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert result.retrieval.documents_considered > 0
    assert result.retrieval.chunks_returned > 0
    # chunks_returned is selected evidence supplied to the generator.
    assert result.retrieval.chunks_returned <= result.retrieval.documents_considered


@pytest.mark.asyncio
async def test_retrieval_improvement_fixes_bad_answer():
    """Better corpus evidence changes a prompt-aware fake from abstention to cited answer.

    Distinguishes retrieval quality from citation-ID validation: the same fake
    and question only succeed after the answer-bearing chunk is indexed.
    """
    sparse = TokenOverlapDocumentStore(
        [
            _chunk(
                "unrelated",
                "Express middleware runs in order for each request.",
                source_id="express-guide",
                framework="express",
                language="typescript",
            )
        ],
        max_limit=10,
    )
    client = PromptAwareClient()
    processor = GroundedQuestionProcessor(
        client,
        sparse,
        budget=_budget(),
        retrieval_top_k=5,
    )

    sparse_result = await processor.process(
        "req-sparse",
        "How does FastAPI inject dependencies with Depends?",
    )
    assert sparse_result.citations == []
    assert sparse_result.answer.startswith("I do not have enough evidence")
    sparse_evidence = client.prompts[0].split("Question:", 1)[0]
    assert "Evidence:" not in sparse_evidence or "Depends" not in sparse_evidence

    rich = _corpus_store()
    rich_client = PromptAwareClient()
    rich_processor = GroundedQuestionProcessor(
        rich_client,
        rich,
        budget=_budget(),
        retrieval_top_k=5,
    )
    rich_result = await rich_processor.process(
        "req-rich",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert rich_result.citations
    assert rich_result.citations[0].chunk_id == "fastapi-deps"
    assert "Depends" in rich_result.answer
    assert "Depends" in rich_client.prompts[0]


@pytest.mark.asyncio
async def test_evidence_framing_includes_selected_chunks_and_question():
    """Assembled prompt uses the budget evidence wrapper and omits non-selected text."""
    store = TokenOverlapDocumentStore(
        [
            _chunk(
                "fastapi-deps",
                "FastAPI dependency injection uses Depends.",
            ),
            _chunk(
                "unrelated-noise",
                "This unrelated noise must not appear when not selected.",
                source_id="noise",
                framework=None,
                language=None,
            ),
        ],
        max_limit=10,
    )
    client = RecordingStructuredClient(_answer("FastAPI uses Depends."))
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=_budget(),
        retrieval_top_k=1,
    )

    await processor.process(
        "req-frame",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert len(client.prompts) == 1
    prompt = client.prompts[0]
    assert "Evidence:" in prompt
    assert "Question:" in prompt
    assert "FastAPI dependency injection uses Depends." in prompt
    assert prompt.rstrip().endswith(
        "How does FastAPI inject dependencies with Depends?"
    ) or "How does FastAPI inject dependencies with Depends?" in prompt


@pytest.mark.asyncio
async def test_grounded_success_logs_omit_chunk_and_prompt_text(caplog):
    """Success logs keep correlation fields but omit corpus and prompt bodies.

    :param caplog: Pytest fixture used to capture structured log records."""
    secret_chunk = _chunk(
        "secret-chunk",
        "SECRET_CORPUS_TOKEN_xyz FastAPI dependency injection uses Depends.",
        source_id="secrets",
    )
    store = TokenOverlapDocumentStore(
        [
            _chunk(
                "fastapi-deps",
                "FastAPI dependency injection uses Depends.",
            ),
            secret_chunk,
        ],
        max_limit=10,
    )
    client = RecordingStructuredClient(
        _answer(
            "FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["fastapi-deps"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=_budget(),
        retrieval_top_k=5,
    )
    caplog.set_level(logging.INFO, logger="app.services.question_processor")

    await processor.process(
        "req-log-ok",
        "How does FastAPI inject dependencies with Depends?",
        request_id="http-log-ok",
    )

    # Secret marker was eligible for the evidence prompt but must not appear in logs.
    assert any("SECRET_CORPUS_TOKEN_xyz" in prompt for prompt in client.prompts)
    formatted = "\n".join(logging.Formatter().format(record) for record in caplog.records)
    assert "SECRET_CORPUS_TOKEN_xyz" not in formatted
    for record in caplog.records:
        assert "SECRET_CORPUS_TOKEN_xyz" not in record.getMessage()
        extras = " ".join(str(value) for value in record.__dict__.values())
        assert "SECRET_CORPUS_TOKEN_xyz" not in extras


@pytest.mark.asyncio
async def test_grounded_failure_logs_omit_prompt_and_corpus_text(caplog):
    """Validation failures log a safe category without prompt or chunk text.

    :param caplog: Pytest fixture used to capture structured log records."""
    store = _corpus_store(
        _chunk(
            "secret-chunk",
            "SECRET_FAILURE_TOKEN_abc must stay out of error logs.",
            source_id="secrets",
        )
    )
    client = RecordingStructuredClient(
        _answer(
            "Bad citations.",
            claims=[
                AnswerClaim(
                    claim="Invented support",
                    citation_chunk_ids=["invented-chunk-id"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=_budget(),
        retrieval_top_k=5,
    )
    caplog.set_level(logging.ERROR, logger="app.services.question_processor")

    with pytest.raises(StructuredOutputError):
        await processor.process(
            "req-log-fail",
            "How does FastAPI inject dependencies with Depends?",
            request_id="http-log-fail",
        )

    formatted = "\n".join(logging.Formatter().format(record) for record in caplog.records)
    assert "SECRET_FAILURE_TOKEN_abc" not in formatted
    for record in caplog.records:
        assert "SECRET_FAILURE_TOKEN_abc" not in record.getMessage()


@pytest.mark.asyncio
async def test_optional_classification_filters_narrow_retrieval():
    """When a classifier returns framework/language filters, search is narrowed."""
    store = TokenOverlapDocumentStore(
        [
            _chunk(
                "fastapi-deps",
                "FastAPI dependency injection uses Depends.",
                framework="fastapi",
                language="python",
            ),
            _chunk(
                "express-middleware",
                "Express middleware Depends-like hooks run in order.",
                source_id="express-guide",
                framework="express",
                language="typescript",
            ),
        ],
        max_limit=10,
    )

    async def classify(question: str) -> ClassificationV1:
        """Return a FastAPI/Python classification for any question.

        :param question: User question text.
        :returns: A classification that requests filtered retrieval."""
        return ClassificationV1(
            intent="answer",
            needs_retrieval=True,
            framework="fastapi",
            language="python",
            confidence=0.9,
            clarifying_question=None,
        )

    client = RecordingStructuredClient(
        _answer(
            "FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["fastapi-deps"],
                )
            ],
        )
    )
    processor = GroundedQuestionProcessor(
        client,
        store,
        budget=_budget(),
        retrieval_top_k=5,
        classifier=classify,
    )

    result = await processor.process(
        "req-filter",
        "How does Depends work?",
        request_id="http-filter",
    )

    assert result.citations
    assert all(citation.source_id == "fastapi-guide" for citation in result.citations)
    assert "Express middleware" not in client.prompts[0]


@pytest.mark.asyncio
async def test_unfiltered_path_works_without_classifier():
    """Without a classifier, retrieval still grounds answers from the full store."""
    client = FakeLLMClient(
        model="fake",
        structured_payload=_answer(
            "FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["fastapi-deps"],
                )
            ],
        ),
    )
    processor = GroundedQuestionProcessor(
        client,
        _corpus_store(),
        budget=_budget(),
        retrieval_top_k=5,
    )

    result = await processor.process(
        "req-nofilter",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert result.answer == "FastAPI uses Depends."
    assert result.citations
    assert result.citations[0].chunk_id == "fastapi-deps"


@pytest.mark.asyncio
async def test_grounded_path_does_not_invoke_tool_loop():
    """Grounded answering uses generate_structured, not the Step 5 tool loop."""
    client = RecordingStructuredClient(_answer("FastAPI uses Depends."))
    processor = GroundedQuestionProcessor(
        client,
        _corpus_store(),
        budget=_budget(),
        retrieval_top_k=5,
    )

    await processor.process(
        "req-no-tools",
        "How does FastAPI inject dependencies with Depends?",
    )

    assert len(client.prompts) == 1
