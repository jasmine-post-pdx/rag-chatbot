"""In-process tests for the public HTTP API."""

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.services.questions import QuestionService


@pytest.fixture
def question_service():
    """Return an in-memory question service for tests.

    Returns:
        A fresh in-memory question service instance.
    """
    return QuestionService()


@pytest.fixture
def client(question_service):
    """Yield a TestClient with the question service dependency overridden.

    Args:
        question_service: The question service to inject into the API.

    Yields:
        A test client configured to use the injected service.
    """
    main.app.dependency_overrides[main.get_question_service] = lambda: question_service
    try:
        with TestClient(main.app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        main.app.dependency_overrides.pop(main.get_question_service, None)


def test_health_and_readiness(client):
    """Verify the health and readiness endpoints return ok statuses."""
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_submit_and_poll_question_returns_answer_and_metadata(client):
    """Verify submitting a question returns a completed result and can be polled."""
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    submitted = response.json()
    assert set(submitted) == {"request_id", "answer", "citations", "retrieval", "model"}
    assert submitted["answer"] == "This is a placeholder answer for: What is RAG?"
    assert submitted["citations"] == []
    assert submitted["retrieval"] == {"documents_considered": 0, "chunks_returned": 0}
    assert submitted["model"]["name"] == "stub"

    polled = client.get(f"/v1/questions/{submitted['request_id']}")
    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "completed"
    assert body["result"]["request_id"] == submitted["request_id"]
    assert body["result"]["answer"] == submitted["answer"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "x" * 4001},
        {"question": 42},
        {"question": None},
    ],
)
def test_submit_rejects_invalid_question_payloads(client, body):
    """Verify invalid question payloads are rejected with 422."""
    response = client.post("/v1/questions", json=body)
    assert response.status_code == 422


def test_submit_rejects_malformed_json(client):
    """Verify malformed JSON bodies are rejected with 422."""
    response = client.post(
        "/v1/questions", content='{"question":', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_poll_unknown_question_is_404(client):
    """Verify polling an unknown question returns 404."""
    response = client.get("/v1/questions/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Question not found"}


def test_submit_storage_failure_returns_generic_500(client):
    """Verify unhandled service errors return a generic 500 response."""
    class FailingQuestionService(QuestionService):
        """Stub service that raises an error when creating a job."""

        def create_job(self, request_id):
            """Raise a runtime error to simulate a storage failure.

            Args:
                request_id: Unique identifier for the job (unused).
            """
            raise RuntimeError("database password and stack details")

    failing_service = FailingQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: failing_service
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "database password" not in response.text
