"""Centralized application settings.

Settings are loaded from environment variables and an optional ``.env`` file.
All values have sensible local-development defaults so the application runs
out of the box, but each value can be overridden per environment.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    Attributes:
        log_level: Python logging level for the application.
        questions_db_path: Path to the SQLite database used for question jobs.
        postgres_host: Hostname of the Postgres vector store.
        postgres_port: Port of the Postgres vector store.
        postgres_user: Postgres username.
        postgres_password: Postgres password.
        postgres_db: Postgres database name.
        embedding_provider: Adapter name for embeddings (e.g., "local", "openai").
        embedding_model: Name or identifier of the embedding model.
        generation_provider: Adapter name for answer generation.
        generation_model: Name or identifier of the generation model.
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

    @property
    def postgres_dsn(self) -> str:
        """Return a SQLAlchemy-style Postgres connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
