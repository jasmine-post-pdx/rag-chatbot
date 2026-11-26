"""CLI contract tests for the reproducible Step 7 chunking experiment command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.chunking_experiment import main


def _write_corpus(raw: Path) -> None:
    """Write a tiny markdown corpus for CLI runs.

    :param raw: Destination directory for ``.md`` sources."""
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "fastapi-guide.md").write_text(
        "---\ntitle: FastAPI Guide\nframework: fastapi\nlanguage: python\n---\n"
        "FastAPI dependency injection wires routers to services with Depends.\n",
        encoding="utf-8",
    )
    (raw / "rag-budget.md").write_text(
        "---\ntitle: RAG Context Budget\nlanguage: python\n---\n"
        "Context budgeting reserves tokens for instructions and a safety margin.\n",
        encoding="utf-8",
    )


def test_cli_writes_json_report_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``main`` with a temp corpus and ``--output`` writes a JSON report.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest fixture for environment isolation."""
    corpus = tmp_path / "raw"
    _write_corpus(corpus)
    output = tmp_path / "results" / "chunking-report.json"
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_COLLECTION", "documents")

    code = main(
        [
            "--path",
            str(corpus),
            "--output",
            str(output),
            "--top-k",
            "3",
        ]
    )

    assert code == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "embedding_model" in payload
    assert "results_by_strategy" in payload or "strategies" in payload
    encoded = json.dumps(payload)
    assert "Depends" not in encoded
    assert "safety margin" not in encoded
    assert '"text"' not in encoded


def test_cli_refuses_destructive_clear_of_production_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """CLI rejects flags that would clear the live ``documents`` collection.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest fixture for environment isolation."""
    corpus = tmp_path / "raw"
    _write_corpus(corpus)
    output = tmp_path / "report.json"
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("VECTOR_COLLECTION", "documents")

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--path",
                str(corpus),
                "--output",
                str(output),
                "--clear-collection",
                "documents",
            ]
        )

    assert excinfo.value.code not in (0, None)
    assert not output.exists()


def test_cli_uses_isolated_experiment_collection_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Experiment output records an isolated/prefixed collection, not bare ``documents``.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest fixture for environment isolation."""
    corpus = tmp_path / "raw"
    _write_corpus(corpus)
    output = tmp_path / "report.json"
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("VECTOR_COLLECTION", "documents")

    code = main(["--path", str(corpus), "--output", str(output)])

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    collection = payload.get("vector_collection") or payload.get("collection")
    assert collection is not None
    assert collection != "documents"
    assert "experiment" in str(collection).casefold() or str(collection).startswith(
        "chunking_"
    )
