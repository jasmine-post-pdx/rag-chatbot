"""Shared fixtures for integration tests."""

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.services.question_workflow import QuestionWorkflow


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
