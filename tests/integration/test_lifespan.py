"""Application lifespan ownership of the question job store."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import main
from app.stores.questions import SQLiteQuestionJobStore


def test_lifespan_shares_and_closes_store(monkeypatch):
    """The workflow and routes share the lifespan-owned store until shutdown."""
    stores = []

    def create_store(**kwargs):
        store = SQLiteQuestionJobStore(**kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(main, "SQLiteQuestionJobStore", create_store)
    with TestClient(main.app) as client:
        workflow = main.app.state.question_workflow
        assert workflow._store is stores[0]
        assert client.get("/ready").json() == {"status": "ready"}
        submitted = client.post("/v1/questions", json={"question": "Hello"}).json()
        assert stores[0].get_job(submitted["request_id"]).result.answer == submitted["answer"]

    assert main.app.state.question_workflow is None
    assert stores[0].is_ready() is False


@pytest.mark.parametrize(
    ("failure_stage", "error_type", "message"),
    [
        ("integrity", sqlite3.DatabaseError, "integrity check failed"),
        ("workflow", RuntimeError, "workflow setup failed"),
    ],
)
def test_lifespan_closes_store_on_partial_startup_failure(
    monkeypatch, failure_stage, error_type, message
):
    """Close the created store if integrity checking or workflow setup fails."""
    stores = []

    class RecordingStore(SQLiteQuestionJobStore):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            stores.append(self)

        def verify_integrity(self):
            if failure_stage == "integrity":
                raise sqlite3.DatabaseError("integrity check failed")
            super().verify_integrity()

    def fail_workflow(**kwargs):
        raise RuntimeError("workflow setup failed")

    monkeypatch.setattr(main, "SQLiteQuestionJobStore", RecordingStore)
    if failure_stage == "workflow":
        monkeypatch.setattr(main, "QuestionWorkflow", fail_workflow)

    with pytest.raises(error_type, match=message), TestClient(main.app):
        pytest.fail("startup unexpectedly succeeded")

    assert len(stores) == 1
    assert stores[0].is_ready() is False
    assert main.app.state.question_workflow is None