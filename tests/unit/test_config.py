import pytest

from interview_coach.config import Settings
from interview_coach.exceptions import ConfigurationError


def test_environment_defaults_use_local_real_models(monkeypatch):
    for name in ("LLM_PROVIDER", "RETRIEVAL_BACKEND", "MAX_INTERVIEW_QUESTIONS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "qwen2.5:3b"
    assert settings.retrieval_backend == "hybrid_ollama"
    assert settings.max_interview_questions == 5


def test_openai_compatible_requires_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ConfigurationError, match="LLM_MODEL"):
        Settings.from_env()


def test_upstash_backend_requires_server_side_credentials(monkeypatch):
    monkeypatch.setenv("SESSION_BACKEND", "upstash_redis")
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    with pytest.raises(ConfigurationError, match="UPSTASH_REDIS_REST_URL"):
        Settings.from_env()


def test_json_object_mode_is_available_for_hosted_models():
    settings = Settings(llm_structured_output_mode="json_object")
    settings.validate()
