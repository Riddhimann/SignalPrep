from __future__ import annotations

import asyncio

import httpx
import pytest

from interview_coach.config import Settings
from interview_coach.exceptions import StructuredOutputError
from interview_coach.llm.gateway import OpenAICompatibleGateway
from interview_coach.schemas import RequirementExtraction


def test_openai_compatible_error_keeps_safe_provider_detail(monkeypatch):
    async def fake_post(*_args, **_kwargs):
        request = httpx.Request("POST", "https://provider.test/chat/completions")
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"message": "Unsupported response format"}},
        )
        raise httpx.HTTPStatusError("bad request", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_provider="openai_compatible",
            llm_model="hosted-model",
            llm_base_url="https://provider.test",
            llm_structured_output_mode="json_object",
            llm_max_retries=0,
        )
    )

    with pytest.raises(StructuredOutputError, match="HTTP 400.*Unsupported response format"):
        asyncio.run(
            gateway.generate_structured(
                "system",
                "user",
                RequirementExtraction,
            )
        )
