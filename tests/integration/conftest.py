"""Shared fixtures for integration tests."""

import os

import pytest
from fastapi.testclient import TestClient

# The app settings object is created at import time and reads `.env`. Keep tests
# isolated from developer settings; preserve a real key only for the explicitly
# opted-in provider smoke test.
test_environment = {
    "GENERATION_PROVIDER": "local",
    "GENERATION_MODEL": "stub",
    "QUESTIONS_DB_PATH": ":memory:",
    "LOG_LEVEL": "INFO",
}
if os.getenv("RUN_OPENAI_SMOKE") != "1":
    test_environment["OPENAI_API_KEY"] = "test-only-placeholder"
os.environ.update(test_environment)

from app.api import main  # noqa: E402
from app.services.question_workflow import QuestionWorkflow  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_lifespan_settings(monkeypatch, tmp_path):
    """Keep the app lifespan on test-local database and provider settings."""
    monkeypatch.setattr(
        main.settings,
        "questions_db_path",
        str(tmp_path / "lifespan.sqlite3"),
    )
    monkeypatch.setattr(main.settings, "generation_provider", "local")
    monkeypatch.setattr(main.settings, "generation_model", "stub")
    monkeypatch.setattr(main.settings, "openai_api_key", None)


@pytest.fixture
def question_store(store_factory):
    """Return a fresh in-memory job store."""
    return store_factory()


@pytest.fixture
def question_workflow(question_store):
    """Return a workflow backed by the in-memory test store."""
    return QuestionWorkflow(question_store)


@pytest.fixture
def client(question_workflow):
    """Yield a test client with the workflow dependency overridden."""
    main.app.dependency_overrides[main.get_question_workflow] = lambda: question_workflow
    try:
        with TestClient(main.app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        main.app.dependency_overrides.pop(main.get_question_workflow, None)
