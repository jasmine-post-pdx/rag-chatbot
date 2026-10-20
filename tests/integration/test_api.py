"""In-process tests for the public HTTP API."""

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.services import questions


@pytest.fixture
def client():
    questions._jobs.clear()
    with TestClient(main.app) as test_client:
        yield test_client
    questions._jobs.clear()


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_submit_and_poll_question_returns_answer_and_metadata(client):
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 202
    submitted = response.json()
    assert set(submitted) == {"request_id", "status", "result"}
    assert submitted["status"] == "pending"
    assert submitted["result"] is None

    polled = client.get(f"/v1/questions/{submitted['request_id']}")
    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "completed"
    assert body["result"]["request_id"] == submitted["request_id"]
    assert body["result"]["answer"] == "This is a placeholder answer for: What is RAG?"
    assert body["result"]["citations"] == []
    assert body["result"]["retrieval"] == {"documents_considered": 0, "chunks_returned": 0}
    assert body["result"]["model"]["name"] == "stub"


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
    response = client.post("/v1/questions", json=body)
    assert response.status_code == 422


def test_submit_rejects_malformed_json(client):
    response = client.post(
        "/v1/questions", content='{"question":', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_poll_unknown_question_is_404(client):
    response = client.get("/v1/questions/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Question not found"}


def test_submit_storage_failure_returns_generic_500(client, monkeypatch):
    def fail_create_job(request_id):
        raise RuntimeError("database password and stack details")

    monkeypatch.setattr(main, "create_job", fail_create_job)
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "database password" not in response.text
