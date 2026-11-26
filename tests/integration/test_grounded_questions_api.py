"""HTTP integration tests for Step 8 retrieval-grounded question answers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main
from app.domain.model_outputs import AnswerClaim, AnswerV1
from app.llm import FakeLLMClient
from app.retrieval.document_store import DocumentChunk
from app.services.question_processor import GroundedQuestionProcessor
from app.services.token_budget import ApproximateTokenCounter, ContextBudgetConfig
from tests.support.overlap_store import TokenOverlapDocumentStore


@pytest.fixture
def configure_generation(monkeypatch, tmp_path):
    """Point the job store at a temporary database and expose a settings helper.

    :param monkeypatch: Pytest fixture used to override settings for one test.
    :param tmp_path: Pytest fixture providing a per-test temporary directory.
    :returns: A function that sets the generation provider and model used at startup."""
    monkeypatch.setattr(main.settings, "questions_db_path", str(tmp_path / "q.db"))

    def configure(provider: str, model: str = "test-model") -> None:
        """Set the generation settings read when the app starts.

        :param provider: Value for ``GENERATION_PROVIDER``.
        :param model: Value for ``GENERATION_MODEL``."""
        monkeypatch.setattr(main.settings, "generation_provider", provider)
        monkeypatch.setattr(main.settings, "generation_model", model)

    return configure


def _budget() -> ContextBudgetConfig:
    """Return a large approximate budget for HTTP grounding fixtures.

    :returns: A ``ContextBudgetConfig`` instance."""
    return ContextBudgetConfig(
        counter=ApproximateTokenCounter(),
        context_window_tokens=128000,
        output_reserve_tokens=64,
        safety_margin_tokens=0,
    )


def _fastapi_store() -> TokenOverlapDocumentStore:
    """Return a small FastAPI Depends corpus for grounded HTTP tests.

    :returns: A ``TokenOverlapDocumentStore`` with one Depends chunk."""
    return TokenOverlapDocumentStore(
        [
            DocumentChunk(
                source_id="fastapi-guide",
                chunk_id="fastapi-deps",
                text=(
                    "FastAPI dependency injection wires routers to services. "
                    "Use Depends to inject shared resources into path operations."
                ),
                framework="fastapi",
                language="python",
            )
        ],
        max_limit=10,
    )


def test_post_questions_returns_citations_for_in_corpus_answer(
    configure_generation, monkeypatch
):
    """POST /v1/questions returns citation rows for a grounded in-corpus answer.

    :param configure_generation: Fixture that configures generation settings and a
        temporary SQLite job database.
    :param monkeypatch: Pytest fixture used to replace the provider factory."""
    configure_generation("openai", "fake-model")
    store = _fastapi_store()
    client = FakeLLMClient(
        model="fake-model",
        structured_payload=AnswerV1(
            answer="FastAPI uses Depends for dependency injection.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["fastapi-deps"],
                )
            ],
            insufficient_evidence=False,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_question_processor",
        lambda provider, model, openai_api_key=None, **_kwargs: GroundedQuestionProcessor(
            client,
            store,
            budget=_budget(),
            retrieval_top_k=5,
        ),
    )

    with TestClient(main.app) as http:
        response = http.post(
            "/v1/questions",
            json={"question": "How does FastAPI inject dependencies with Depends?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "question_id",
        "answer",
        "citations",
        "retrieval",
        "model",
    }
    assert body["answer"] == "FastAPI uses Depends for dependency injection."
    assert body["citations"] == [
        {
            "source_id": "fastapi-guide",
            "chunk_id": "fastapi-deps",
            "text": (
                "FastAPI dependency injection wires routers to services. "
                "Use Depends to inject shared resources into path operations."
            ),
        }
    ]
    assert body["retrieval"]["documents_considered"] >= 1
    assert body["retrieval"]["chunks_returned"] >= 1
    assert body["model"]["provider"] == "fake"
    assert body["model"]["name"] == "fake-model"


def test_out_of_corpus_abstention_returns_empty_citations(
    configure_generation, monkeypatch
):
    """Out-of-corpus abstention completes with empty citations and stable shape.

    :param configure_generation: Fixture that configures generation settings and a
        temporary SQLite job database.
    :param monkeypatch: Pytest fixture used to replace the provider factory."""
    configure_generation("openai", "fake-model")
    store = TokenOverlapDocumentStore([], max_limit=5)
    client = FakeLLMClient(
        model="fake-model",
        structured_payload=AnswerV1(
            answer="I do not have enough evidence to answer.",
            claims=[],
            insufficient_evidence=True,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_question_processor",
        lambda provider, model, openai_api_key=None, **_kwargs: GroundedQuestionProcessor(
            client,
            store,
            budget=_budget(),
            retrieval_top_k=5,
        ),
    )

    with TestClient(main.app) as http:
        response = http.post(
            "/v1/questions",
            json={"question": "What is the capital of Atlantis?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "question_id",
        "answer",
        "citations",
        "retrieval",
        "model",
    }
    assert body["citations"] == []
    assert body["answer"] == "I do not have enough evidence to answer."


def test_poll_persists_citation_bearing_result(configure_generation, monkeypatch):
    """GET /v1/questions/{id} returns the same citation-bearing completed result.

    :param configure_generation: Fixture that configures generation settings and a
        temporary SQLite job database.
    :param monkeypatch: Pytest fixture used to replace the provider factory."""
    configure_generation("openai", "fake-model")
    store = _fastapi_store()
    client = FakeLLMClient(
        model="fake-model",
        structured_payload=AnswerV1(
            answer="FastAPI uses Depends.",
            claims=[
                AnswerClaim(
                    claim="Depends injects shared resources",
                    citation_chunk_ids=["fastapi-deps"],
                )
            ],
            insufficient_evidence=False,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_question_processor",
        lambda provider, model, openai_api_key=None, **_kwargs: GroundedQuestionProcessor(
            client,
            store,
            budget=_budget(),
            retrieval_top_k=5,
        ),
    )

    with TestClient(main.app) as http:
        submitted = http.post(
            "/v1/questions",
            json={"question": "How does FastAPI inject dependencies with Depends?"},
        )
        assert submitted.status_code == 200
        body = submitted.json()
        question_id = body["question_id"]
        polled = http.get(f"/v1/questions/{question_id}")

    assert polled.status_code == 200
    assert polled.json() == {
        "question_id": question_id,
        "status": "completed",
        "result": body,
        "error_category": None,
    }
    assert polled.json()["result"]["citations"]
    assert polled.json()["result"]["retrieval"]["chunks_returned"] >= 1
