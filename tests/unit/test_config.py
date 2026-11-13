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
    """Keep timeout, retries, output cap, and instructions at today's behavior.

    :param monkeypatch: Pytest fixture used to clear environment variables."""
    for name in (
        "GENERATION_TIMEOUT_SECONDS",
        "GENERATION_MAX_RETRIES",
        "GENERATION_MAX_OUTPUT_TOKENS",
        "GENERATION_INSTRUCTIONS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.generation_timeout_seconds == 30.0
    assert settings.generation_max_retries == 2
    assert settings.generation_max_output_tokens is None
    assert settings.generation_instructions is None


def test_generation_client_settings_load_from_environment(monkeypatch):
    """Load generation client overrides from the environment.

    :param monkeypatch: Pytest fixture used to set environment variables."""
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("GENERATION_MAX_RETRIES", "4")
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("GENERATION_INSTRUCTIONS", "  Be concise.  ")

    settings = Settings(_env_file=None)

    assert settings.generation_timeout_seconds == 12.5
    assert settings.generation_max_retries == 4
    assert settings.generation_max_output_tokens == 256
    assert settings.generation_instructions == "Be concise."


@pytest.mark.parametrize(
    ("env_name", "env_value", "match"),
    [
        ("GENERATION_TIMEOUT_SECONDS", "0", "greater than 0"),
        ("GENERATION_MAX_RETRIES", "-1", "greater than or equal to 0"),
        ("GENERATION_MAX_OUTPUT_TOKENS", "0", "greater than 0"),
        ("GENERATION_INSTRUCTIONS", "   ", "GENERATION_INSTRUCTIONS must not be empty"),
    ],
)
def test_generation_client_settings_reject_invalid_values(
    monkeypatch, env_name, env_value, match
):
    """Reject timeouts, retry counts, output caps, and instructions that cannot be used.

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
