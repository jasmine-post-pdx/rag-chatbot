"""Opt-in smoke test for the real OpenAI Responses API integration."""

import os

import pytest

from app.llm.generation import OpenAIResponsesClient


@pytest.mark.openai_smoke
@pytest.mark.asyncio
async def test_openai_responses_api_smoke():
    """Verify a real async Responses API call returns answer and usage metadata.

    Set ``RUN_OPENAI_SMOKE=1``, ``OPENAI_API_KEY``, and
    ``OPENAI_SMOKE_MODEL`` to opt in. The key and model can be set in the shell
    or ``.env`` file. Without the opt-in flag, the test skips before loading
    credentials or making a network request.
    """
    if os.getenv("RUN_OPENAI_SMOKE") != "1":
        pytest.skip("Set RUN_OPENAI_SMOKE=1 to enable the real-provider smoke test")

    from app.config.settings import Settings

    settings = Settings()
    if settings.openai_api_key is None:
        pytest.fail("OPENAI_API_KEY must be configured to run the OpenAI smoke test")

    model = os.getenv("OPENAI_SMOKE_MODEL")
    if not model:
        from dotenv import dotenv_values

        model = dotenv_values(".env").get("OPENAI_SMOKE_MODEL")
    if not model:
        pytest.fail("OPENAI_SMOKE_MODEL must be configured to run the OpenAI smoke test")

    client = OpenAIResponsesClient(
        model=model,
        api_key=settings.openai_api_key.get_secret_value(),
        timeout_seconds=settings.generation_timeout_seconds,
        max_output_tokens=settings.generation_max_output_tokens,
        instructions=settings.generation_instructions,
        max_retries=settings.generation_max_retries,
    )
    result = await client.generate("Reply with a short greeting.")

    assert isinstance(result.answer, str)
    assert result.answer.strip()
    assert len(result.answer) < 8_000
    assert result.provider == "openai"
    assert isinstance(result.model, str) and result.model
    assert result.finish_status in {None, "completed", "incomplete"}
    assert result.refusal is None
    assert result.provider_request_id is None or (
        isinstance(result.provider_request_id, str) and result.provider_request_id
    )
    assert result.input_tokens is None or result.input_tokens >= 0
    assert result.output_tokens is None or result.output_tokens >= 0
