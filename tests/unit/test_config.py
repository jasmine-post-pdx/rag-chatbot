"""Unit tests for centralized application settings."""

import pytest
from pydantic import SecretStr, ValidationError

from app.config.settings import Settings


def test_postgres_dsn_encodes_special_characters():
    """Verify credentials with special characters are URL-encoded."""
    settings = Settings(
        postgres_user="raglab",
        postgres_password="p@ss:w0rd/",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="raglab",
    )

    dsn = settings.postgres_dsn

    assert dsn == (
        "postgresql://raglab:p%40ss%3Aw0rd%2F@localhost:5432/raglab"
    )


def test_postgres_dsn_default_values():
    """Verify the DSN uses default local-development values."""
    settings = Settings()

    assert settings.postgres_dsn == (
        "postgresql://raglab:raglab@localhost:5432/raglab"
    )


@pytest.mark.parametrize("api_key", ["", "   ", "\t\n"])
def test_empty_environment_api_key_fails_settings_validation(monkeypatch, api_key):
    """Reject an empty or whitespace-only API key loaded from the environment.

    :param monkeypatch: Pytest fixture used to set environment variables.
    :param api_key: Invalid empty value to validate."""
    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    with pytest.raises(ValidationError, match="OPENAI_API_KEY must not be empty"):
        Settings(_env_file=None)


def test_generation_client_settings_use_current_client_defaults(monkeypatch):
    """Keep timeout, retries, budget, output cap, and instructions defaults.

    :param monkeypatch: Pytest fixture used to clear environment variables."""
    for name in (
        "GENERATION_TIMEOUT_SECONDS",
        "GENERATION_MAX_RETRIES",
        "GENERATION_MAX_OUTPUT_TOKENS",
        "GENERATION_CONTEXT_WINDOW_TOKENS",
        "GENERATION_BUDGET_SAFETY_MARGIN_TOKENS",
        "GENERATION_BUDGET_DEBUG",
        "GENERATION_INSTRUCTIONS",
        "TOOL_SEARCH_MAX_LIMIT",
        "TOOL_CALL_TIMEOUT_SECONDS",
        "TOOL_MAX_RESULT_CHARS",
        "TOOL_MAX_TURNS",
        "TOOL_MAX_CALLS",
        "VECTOR_STORE",
        "VECTOR_COLLECTION",
        "SIMILARITY_METRIC",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "SEARCH_DEFAULT_TOP_K",
        "SEARCH_MAX_TOP_K",
        "EMBEDDING_DIMENSIONS",
        "CORPUS_RAW_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.generation_timeout_seconds == 30.0
    assert settings.generation_max_retries == 2
    assert settings.generation_max_output_tokens == 1024
    assert settings.generation_context_window_tokens == 128000
    assert settings.generation_budget_safety_margin_tokens == 256
    assert settings.generation_budget_debug is False
    assert settings.generation_instructions is None
    assert settings.tool_search_max_limit == 5
    assert settings.tool_call_timeout_seconds == 2.0
    assert settings.tool_max_result_chars == 4000
    assert settings.tool_max_turns == 3
    assert settings.tool_max_calls == 6
    assert settings.vector_store == "memory"
    assert settings.vector_collection == "documents"
    assert settings.similarity_metric == "cosine"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.search_default_top_k == 5
    assert settings.search_max_top_k == 20
    assert settings.embedding_dimensions == 32
    assert settings.corpus_raw_dir == "data/raw"
    assert settings.embedding_provider == "fake"
    assert settings.embedding_model == "fake-embed"


def test_generation_client_settings_load_from_environment(monkeypatch):
    """Load generation client overrides from the environment.

    :param monkeypatch: Pytest fixture used to set environment variables."""
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("GENERATION_MAX_RETRIES", "4")
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("GENERATION_CONTEXT_WINDOW_TOKENS", "8192")
    monkeypatch.setenv("GENERATION_BUDGET_SAFETY_MARGIN_TOKENS", "64")
    monkeypatch.setenv("GENERATION_BUDGET_DEBUG", "true")
    monkeypatch.setenv("GENERATION_INSTRUCTIONS", "  Be concise.  ")
    monkeypatch.setenv("TOOL_SEARCH_MAX_LIMIT", "3")
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("TOOL_MAX_RESULT_CHARS", "1000")
    monkeypatch.setenv("TOOL_MAX_TURNS", "2")
    monkeypatch.setenv("TOOL_MAX_CALLS", "4")

    settings = Settings(_env_file=None)

    assert settings.generation_timeout_seconds == 12.5
    assert settings.generation_max_retries == 4
    assert settings.generation_max_output_tokens == 256
    assert settings.generation_context_window_tokens == 8192
    assert settings.generation_budget_safety_margin_tokens == 64
    assert settings.generation_budget_debug is True
    assert settings.generation_instructions == "Be concise."
    assert settings.tool_search_max_limit == 3
    assert settings.tool_call_timeout_seconds == 1.5
    assert settings.tool_max_result_chars == 1000
    assert settings.tool_max_turns == 2
    assert settings.tool_max_calls == 4


@pytest.mark.parametrize(
    ("env_name", "env_value", "match"),
    [
        ("GENERATION_TIMEOUT_SECONDS", "0", "greater than 0"),
        ("GENERATION_MAX_RETRIES", "-1", "greater than or equal to 0"),
        ("GENERATION_MAX_OUTPUT_TOKENS", "0", "greater than 0"),
        ("GENERATION_CONTEXT_WINDOW_TOKENS", "0", "greater than 0"),
        ("GENERATION_BUDGET_SAFETY_MARGIN_TOKENS", "-1", "greater than or equal to 0"),
        ("GENERATION_INSTRUCTIONS", "   ", "GENERATION_INSTRUCTIONS must not be empty"),
        ("TOOL_SEARCH_MAX_LIMIT", "0", "greater than or equal to 1"),
        ("TOOL_CALL_TIMEOUT_SECONDS", "0", "greater than 0"),
        ("TOOL_MAX_RESULT_CHARS", "0", "greater than or equal to 1"),
        ("TOOL_MAX_TURNS", "0", "greater than or equal to 1"),
        ("TOOL_MAX_CALLS", "0", "greater than or equal to 1"),
        ("VECTOR_STORE", "qdrant", "memory' or 'pgvector"),
        ("SIMILARITY_METRIC", "dot", "cosine"),
        ("CHUNK_SIZE", "0", "greater than 0"),
        ("SEARCH_DEFAULT_TOP_K", "0", "greater than or equal to 1"),
        ("SEARCH_MAX_TOP_K", "0", "greater than or equal to 1"),
        ("EMBEDDING_DIMENSIONS", "0", "greater than 0"),
    ],
)
def test_generation_client_settings_reject_invalid_values(
    monkeypatch, env_name, env_value, match
):
    """Reject timeouts, retry counts, budget, output caps, and instructions that cannot be used.

    :param monkeypatch: Pytest fixture used to set environment variables.
    :param env_name: Setting environment variable to override.
    :param env_value: Invalid value to validate.
    :param match: Expected validation message fragment."""
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValidationError, match=match):
        Settings(_env_file=None)


def test_environment_api_key_is_loaded_as_secret(monkeypatch):
    """Load a non-empty environment API key into a masked secret setting.

    :param monkeypatch: Pytest fixture used to set environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.openai_api_key) == "**********"
