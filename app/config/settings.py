"""Centralized application settings.

Settings are loaded from environment variables and an optional ``.env`` file.
All values have sensible local-development defaults so the application runs
out of the box, but each value can be overridden per environment.
"""

from urllib.parse import quote

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    Attributes:
        log_level: Python logging level for the application.
        questions_db_path: Path to the SQLite database used for question jobs.
        questions_archive_dir: Directory for monthly terminal-job archives.
        questions_backup_dir: Directory for SQLite backup snapshots.
        questions_live_retention_days: Days jobs and idempotency keys stay live.
        questions_archive_retention_days: Days archived jobs are retained.
        postgres_host: Hostname of the Postgres vector store.
        postgres_port: Port of the Postgres vector store.
        postgres_user: Postgres username.
        postgres_password: Postgres password.
        postgres_db: Postgres database name.
        embedding_provider: Adapter name for embeddings (e.g., "local", "openai").
        embedding_model: Name or identifier of the embedding model.
        generation_provider: Adapter name for answer generation.
        generation_model: Name or identifier of the generation model.
        generation_timeout_seconds: Per-request generation client timeout.
        generation_max_retries: Extra attempts after a failed generation request.
        generation_max_output_tokens: Optional completion-token cap. Unset means
            no explicit cap.
        generation_instructions: Optional instructions sent separately from the
            question. Unset means no extra instructions.
        openai_api_key: Optional OpenAI API key loaded from the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = "INFO"

    # Job store
    questions_db_path: str = "data/questions.db"
    questions_archive_dir: str = "data/question_archives"
    questions_backup_dir: str = "data/question_backups"
    questions_live_retention_days: int = Field(default=30, gt=0)
    questions_archive_retention_days: int = Field(default=365, gt=0)

    # Vector store (used once retrieval is wired up)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "raglab"
    postgres_password: str = "raglab"
    postgres_db: str = "raglab"

    # Model adapters
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_provider: str = "local"
    generation_model: str = "stub"
    generation_timeout_seconds: float = Field(default=30.0, gt=0)
    generation_max_retries: int = Field(default=2, ge=0)
    generation_max_output_tokens: int | None = Field(default=None, gt=0)
    generation_instructions: str | None = None
    openai_api_key: SecretStr | None = None

    @field_validator("generation_instructions", mode="before")
    @classmethod
    def strip_generation_instructions(cls, value: object) -> object:
        """Reject a blank instructions value and strip surrounding whitespace."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("GENERATION_INSTRUCTIONS must not be empty")
            return stripped
        return value

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def reject_empty_openai_api_key(cls, value: object) -> object:
        """Reject an explicitly configured empty or whitespace-only key."""
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("OPENAI_API_KEY must not be empty")
            return value.strip()
        return value

    @property
    def postgres_dsn(self) -> str:
        """Return a SQLAlchemy-style Postgres connection URL."""
        return (
            f"postgresql://{quote(self.postgres_user, safe='')}"
            f":{quote(self.postgres_password, safe='')}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{quote(self.postgres_db, safe='')}"
        )


settings = Settings()
