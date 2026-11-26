"""Settings coverage for the Step 7 ``chunking_strategy`` configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_chunking_strategy_defaults_to_recursive_character(monkeypatch):
    """Unset ``CHUNKING_STRATEGY`` selects recursive character splitting.

    :param monkeypatch: Pytest fixture used to clear environment variables."""
    monkeypatch.delenv("CHUNKING_STRATEGY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.chunking_strategy == "recursive_character"


@pytest.mark.parametrize(
    "value",
    ["recursive_character", "token_aware", "structure_aware", "TOKEN_AWARE"],
)
def test_chunking_strategy_accepts_known_variants(monkeypatch, value: str):
    """Known strategy names load from the environment (case-insensitive).

    :param monkeypatch: Pytest fixture used to set environment variables.
    :param value: Supported strategy name."""
    monkeypatch.setenv("CHUNKING_STRATEGY", value)

    settings = Settings(_env_file=None)

    assert settings.chunking_strategy == value.casefold()


@pytest.mark.parametrize(
    "value",
    ["qdrant", "semantic", "", "  "],
)
def test_chunking_strategy_rejects_unknown_or_blank_values(monkeypatch, value: str):
    """Unknown or blank strategy names fail settings validation.

    :param monkeypatch: Pytest fixture used to set environment variables.
    :param value: Invalid strategy name."""
    monkeypatch.setenv("CHUNKING_STRATEGY", value)

    with pytest.raises(ValidationError, match="chunking_strategy|CHUNKING_STRATEGY"):
        Settings(_env_file=None)
