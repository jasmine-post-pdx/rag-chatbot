"""Unit tests for centralized application settings."""

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
