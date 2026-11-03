"""Unit tests for centralized application settings."""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


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

    Args:
        monkeypatch: Pytest fixture used to set environment variables.
        api_key: Invalid empty value to validate.
    """
    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    with pytest.raises(ValidationError, match="OPENAI_API_KEY must not be empty"):
        Settings(_env_file=None)


def test_environment_api_key_is_loaded_as_secret(monkeypatch):
    """Load a non-empty environment API key into a masked secret setting.

    Args:
        monkeypatch: Pytest fixture used to set environment variables.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert str(settings.openai_api_key) == "**********"
