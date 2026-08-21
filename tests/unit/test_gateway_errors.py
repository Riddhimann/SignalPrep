from __future__ import annotations

import asyncio
import json

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


def test_hosted_qwen_disables_reasoning_for_json_generation(monkeypatch):
    captured: dict = {}

    async def fake_post(_self, *_args, **kwargs):
        captured.update(kwargs["json"])
        request = httpx.Request("POST", "https://provider.test/chat/completions")
        content = RequirementExtraction(
            target_role="Data Scientist",
            responsibilities=["Build models"],
            required_skills=["Python"],
            preferred_skills=[],
            evaluation_topics=["model validation"],
        ).model_dump_json()
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_provider="openai_compatible",
            llm_model="qwen/qwen3.6-27b",
            llm_base_url="https://provider.test",
            llm_structured_output_mode="json_object",
        )
    )
    result = asyncio.run(
        gateway.generate_structured("system", json.dumps({"input": "test"}), RequirementExtraction)
    )

    assert result.required_skills == ["Python"]
    assert captured["reasoning_effort"] == "none"
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured
