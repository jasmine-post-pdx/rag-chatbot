"""In-process tests for the public HTTP API."""

import logging
from typing import ClassVar

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
    ("body", "expected_substring"),
    [
        ({}, "question: Field required"),
        ({"question": ""}, "String should have at least 1 character"),
        ({"question": "x" * 4001}, "String should have at most 4000 characters"),
        ({"question": 42}, "Input should be a valid string"),
        ({"question": None}, "Input should be a valid string"),
    ],
)
def test_submit_rejects_invalid_question_payloads(client, body, expected_substring):
    """Verify invalid question payloads are rejected with a structured 422 response."""
    response = client.post("/v1/questions", json=body)

    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"detail"}
    assert body["detail"].startswith("Invalid request:")
    assert expected_substring in body["detail"]


def test_submit_rejects_malformed_json(client):
    """Verify malformed JSON bodies are rejected with a structured 422 response."""
    response = client.post(
        "/v1/questions", content='{"question":', headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"detail"}
    assert body["detail"].startswith("Invalid request:")
    assert "JSON" in body["detail"]


def test_poll_unknown_question_is_404(client):
    """Verify polling an unknown question returns 404."""
    response = client.get("/v1/questions/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Question not found"}


def test_invalid_server_response_data_returns_500(client):
    """Verify response-validation failures for server-generated data return 500, not 4xx."""

    class BadResponseQuestionService(QuestionService):
        """Stub service that returns malformed response data."""

        def get_job(self, request_id):
            """Return a fake job whose result violates the response model."""

            class FakeJob:
                result: ClassVar[dict] = {
                    "request_id": request_id,
                    "answer": 123,
                    "citations": "not a list",
                    "retrieval": {
                        "documents_considered": "x",
                        "chunks_returned": "y",
                    },
                    "model": {
                        "provider": "stub",
                        "name": "stub",
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                }

            return FakeJob()

    bad_service = BadResponseQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: bad_service
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


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


def _find_request_records(caplog):
    """Return records emitted by the request logging middleware."""
    return [r for r in caplog.records if r.name == "app.api.middleware"]


def test_request_log_includes_context(client, caplog):
    """Verify successful requests log method, route, status, timing, and ID."""
    caplog.set_level(logging.INFO)
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    records = _find_request_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.method == "POST"
    assert record.route == "/v1/questions"
    assert record.status == 200
    assert record.elapsed_ms >= 0
    assert record.request_id


def test_request_id_header_is_returned(client):
    """Verify each response carries a generated correlation ID."""
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert response.headers["x-request-id"]


def test_request_id_header_is_preserved(client, caplog):
    """Verify an incoming correlation ID is kept and logged."""
    caplog.set_level(logging.INFO)
    response = client.post(
        "/v1/questions",
        json={"question": "What is RAG?"},
        headers={"x-request-id": "existing-request-id"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "existing-request-id"
    records = _find_request_records(caplog)
    assert records[0].request_id == "existing-request-id"


def test_request_log_does_not_contain_question_text(client, caplog):
    """Verify raw request bodies are not written to logs."""
    caplog.set_level(logging.INFO)
    response = client.post("/v1/questions", json={"question": "secret question text"})

    assert response.status_code == 200
    assert "secret question text" not in caplog.text


def test_validation_error_logs_category(client, caplog):
    """Verify validation failures include the validation error category."""
    caplog.set_level(logging.INFO)
    response = client.post("/v1/questions", json={})

    assert response.status_code == 422
    records = _find_request_records(caplog)
    assert records[0].status == 422
    assert records[0].error_category == "validation"


def test_server_error_logs_category(client, caplog):
    """Verify unhandled service errors include the server error category."""
    caplog.set_level(logging.INFO)

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
    records = _find_request_records(caplog)
    assert records[0].status == 500
    assert records[0].error_category == "server"
