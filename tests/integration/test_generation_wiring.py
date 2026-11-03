"""Integration tests for wiring the configured generation provider at startup."""

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.models import FakeLLMClient
from app.services.question_answerer import QuestionAnswerer
from app.services.processor_factory import UnknownGenerationProviderError


@pytest.fixture
def configure_generation(monkeypatch, tmp_path):
    """Point the job store at a temporary database and expose a settings helper.

    Args:
        monkeypatch: Pytest fixture used to override settings for one test.
        tmp_path: Pytest fixture providing a per-test temporary directory.

    Returns:
        A function that sets the generation provider and model used at startup.
    """
    monkeypatch.setattr(main.settings, "questions_db_path", str(tmp_path / "q.db"))

    def configure(provider: str, model: str = "test-model") -> None:
        """Set the generation settings read when the app starts.

        Args:
            provider: Value for ``GENERATION_PROVIDER``.
            model: Value for ``GENERATION_MODEL``.
        """
        monkeypatch.setattr(main.settings, "generation_provider", provider)
        monkeypatch.setattr(main.settings, "generation_model", model)

    return configure


def test_local_provider_serves_placeholder_answers(configure_generation):
    """Verify the default local provider keeps returning placeholder answers."""
    configure_generation("local")

    with TestClient(main.app) as client:
        response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a placeholder answer for: What is RAG?"


def test_fake_answerer_returns_and_persists_api_result(
    configure_generation, monkeypatch
):
    """Verify the API returns a fake-generated answer and persists its job.

    Args:
        configure_generation: Fixture that configures generation settings and a
            temporary SQLite job database.
        monkeypatch: Pytest fixture used to replace the provider factory.
    """
    configure_generation("openai", "fake-model")
    monkeypatch.setattr(
        main,
        "build_question_processor",
        lambda provider, model, openai_api_key=None: QuestionAnswerer(
            FakeLLMClient(model=model)
        ),
    )

    with TestClient(main.app) as client:
        response = client.post("/v1/questions", json={"question": "What is RAG?"})
        assert response.status_code == 200
        submitted = response.json()

        request_id = submitted["request_id"]
        polled = client.get(f"/v1/questions/{request_id}")

    assert submitted == {
        "request_id": request_id,
        "answer": "Fake answer to: What is RAG?",
        "citations": [],
        "retrieval": {"documents_considered": 0, "chunks_returned": 0},
        "model": {
            "provider": "fake",
            "name": "fake-model",
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }
    assert polled.status_code == 200
    assert polled.json() == {
        "request_id": request_id,
        "status": "completed",
        "result": submitted,
        "error_category": None,
    }


def test_unknown_provider_fails_at_startup(configure_generation):
    """Verify an unsupported provider is rejected when the app starts."""
    configure_generation("not-a-provider")

    with pytest.raises(UnknownGenerationProviderError), TestClient(main.app):
        pass


def test_openai_provider_without_api_key_starts_and_fails_on_use(
    configure_generation, monkeypatch, caplog
):
    """Verify a missing API key only fails, with a clear log, when a question is asked."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    configure_generation("openai", "gpt-test")

    with TestClient(main.app, raise_server_exceptions=False) as client:
        assert client.get("/ready").status_code == 200

        response = client.post("/v1/questions", json={"question": "What is RAG?"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}

        question_id = response.headers["x-question-id"]
        job = client.get(f"/v1/questions/{question_id}").json()

    assert job["status"] == "failed"
    assert job["error_category"] == "processing_error"
    assert "OPENAI_API_KEY is not set" in caplog.text
