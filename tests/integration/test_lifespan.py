"""Application lifespan ownership of the question job store."""

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.retrieval import factory as retrieval_factory
from app.stores.sqlite import SQLiteQuestionJobStore


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
        assert main.app.state.question_job_store is stores[0]
        assert main.app.state.generation_client is None
        assert client.get("/ready").json() == {"status": "ready"}
        submitted = client.post("/v1/questions", json={"question": "Hello"}).json()
        assert stores[0].get_job(submitted["question_id"]).result.answer == submitted["answer"]

    assert main.app.state.question_workflow is None
    assert main.app.state.question_job_store is None
    assert main.app.state.generation_client is None
    assert stores[0].is_ready() is False


def test_lifespan_closes_shared_openai_client(monkeypatch, tmp_path):
    """Create and close a shared AsyncOpenAI client when OpenAI is configured."""
    import openai as openai_module

    closed = []

    class RecordingAsyncOpenAI:
        """Record close calls for the lifespan-owned generation client."""

        def __init__(self, **kwargs):
            """Store construction kwargs for assertions.

            :param **kwargs: Arguments passed by the lifespan factory."""
            self.kwargs = kwargs

        async def close(self) -> None:
            """Mark the shared client as closed."""
            closed.append(True)

    monkeypatch.setattr(main.settings, "questions_db_path", str(tmp_path / "q.db"))
    monkeypatch.setattr(main.settings, "generation_provider", "openai")
    monkeypatch.setattr(main.settings, "generation_model", "gpt-test")
    monkeypatch.setattr(
        main.settings,
        "openai_api_key",
        SimpleNamespace(get_secret_value=lambda: "test-key"),
    )
    monkeypatch.setattr(openai_module, "AsyncOpenAI", RecordingAsyncOpenAI)

    with TestClient(main.app) as client:
        shared = main.app.state.generation_client
        assert isinstance(shared, RecordingAsyncOpenAI)
        assert shared.kwargs == {
            "api_key": "test-key",
            "timeout": main.settings.generation_timeout_seconds,
            "max_retries": 0,
        }
        assert client.get("/ready").status_code == 200

    assert closed == [True]
    assert main.app.state.generation_client is None


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
    assert main.app.state.question_job_store is None
    assert main.app.state.question_workflow is None

def test_lifespan_closes_embedding_client(monkeypatch, tmp_path):
    """Shutdown awaits ``aclose`` on the lifespan-owned embedding client."""
    closed: list[bool] = []

    class RecordingEmbedder:
        """Minimal embedder that records ``aclose`` for lifespan cleanup."""

        model = "fake-lifespan"
        dimensions = 8

        def embed_documents(self, texts):
            """Return zero vectors sized to ``dimensions``.

            :param texts: Document texts.
            :returns: Zero embeddings."""
            return [[0.0] * self.dimensions for _ in texts]

        def embed_query(self, text):
            """Return a zero query vector.

            :param text: Query text.
            :returns: Zero embedding."""
            return [0.0] * self.dimensions

        async def aclose(self) -> None:
            """Record that lifespan closed the embedding client."""
            closed.append(True)

    monkeypatch.setattr(main.settings, "questions_db_path", str(tmp_path / "q.db"))
    monkeypatch.setattr(main.settings, "corpus_raw_dir", str(tmp_path / "missing-raw"))
    monkeypatch.setattr(main.settings, "generation_provider", "local")
    monkeypatch.setattr(main, "build_embedding_client", lambda settings: RecordingEmbedder())

    with TestClient(main.app) as client:
        assert main.app.state.embedding_client is not None
        assert client.get("/ready").status_code == 200

    assert closed == [True]
    assert main.app.state.embedding_client is None


def test_lifespan_seeds_via_async_bounded_helper(monkeypatch, tmp_path):
    """Startup awaits ``async_seed_corpus_if_present`` instead of sync seed on the loop."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fastapi-guide.md").write_text(
        "FastAPI Depends injects shared resources.\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    async def recording_async_seed(settings, store, embedder):
        """Record that the async seed helper ran.

        :param settings: Application settings.
        :param store: Vector store.
        :param embedder: Embedding client."""
        calls.append("async")

    def recording_sync_seed(settings, store, embedder):
        """Record that the sync seed path still ran on the event loop.

        :param settings: Application settings.
        :param store: Vector store.
        :param embedder: Embedding client."""
        calls.append("sync")

    monkeypatch.setattr(main.settings, "questions_db_path", str(tmp_path / "q.db"))
    monkeypatch.setattr(main.settings, "corpus_raw_dir", str(raw))
    monkeypatch.setattr(main.settings, "generation_provider", "local")
    monkeypatch.setattr(main, "async_seed_corpus_if_present", recording_async_seed)
    monkeypatch.setattr(
        retrieval_factory, "async_seed_corpus_if_present", recording_async_seed
    )
    monkeypatch.setattr(
        retrieval_factory, "seed_corpus_if_present", recording_sync_seed
    )

    with TestClient(main.app) as client:
        assert client.get("/ready").status_code == 200

    assert calls == ["async"]
