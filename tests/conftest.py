"""Shared test fixtures."""

import pytest

from app.stores.sqlite_question_jobs import SQLiteQuestionJobStore


@pytest.fixture
def store_factory():
    """Create SQLite stores and close each one after the test."""
    stores = []

    def create_store(*args, store_type=SQLiteQuestionJobStore, **kwargs):
        store = store_type(*args, **kwargs)
        stores.append(store)
        return store

    yield create_store

    for store in reversed(stores):
        store.close()
