from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from interview_coach.exceptions import ConfigurationError


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {value!r}")


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    llm_provider: str = "deterministic_demo"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: int = 30
    llm_max_tokens: int = 1800
    llm_seed: int = 42
    llm_max_retries: int = 2
    llm_structured_output_mode: str = "json_schema"
    llm_reasoning_effort: str = ""
    retrieval_backend: str = "lexical_demo"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    stt_model_size: str = "small"
    speech_emotion_model: str = ""
    text_emotion_model: str = ""
    max_interview_questions: int = 5
    max_document_bytes: int = 5 * 1024 * 1024
    max_audio_bytes: int = 25 * 1024 * 1024
    max_audio_seconds: int = 300
    data_retention: bool = False
    session_backend: str = "memory"
    session_ttl_seconds: int = 24 * 60 * 60
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        provider = os.getenv("LLM_PROVIDER", "ollama").strip()
        default_model = "qwen2.5:3b" if provider == "ollama" else ""
        default_base_url = "http://127.0.0.1:11434" if provider == "ollama" else ""
        settings = cls(
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", default_model).strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", default_base_url).strip(),
            llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 120),
            llm_max_tokens=_int("LLM_MAX_TOKENS", 1800),
            llm_seed=_int("LLM_SEED", 42, minimum=0),
            llm_max_retries=_int("LLM_MAX_RETRIES", 2, minimum=0),
            llm_structured_output_mode=os.getenv(
                "LLM_STRUCTURED_OUTPUT_MODE", "json_schema"
            ).strip(),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "").strip(),
            retrieval_backend=os.getenv("RETRIEVAL_BACKEND", "hybrid_ollama").strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest",
            ).strip(),
            stt_model_size=os.getenv("STT_MODEL_SIZE", "small").strip(),
            speech_emotion_model=os.getenv("SPEECH_EMOTION_MODEL", "").strip(),
            text_emotion_model=os.getenv("TEXT_EMOTION_MODEL", "").strip(),
            max_interview_questions=_int("MAX_INTERVIEW_QUESTIONS", 5),
            max_document_bytes=_int("MAX_DOCUMENT_BYTES", 5 * 1024 * 1024),
            max_audio_bytes=_int("MAX_AUDIO_BYTES", 25 * 1024 * 1024),
            max_audio_seconds=_int("MAX_AUDIO_SECONDS", 300),
            data_retention=_bool("DATA_RETENTION", False),
            session_backend=os.getenv("SESSION_BACKEND", "memory").strip(),
            session_ttl_seconds=_int("SESSION_TTL_SECONDS", 24 * 60 * 60),
            upstash_redis_rest_url=os.getenv("UPSTASH_REDIS_REST_URL", "").strip(),
            upstash_redis_rest_token=os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.llm_provider not in {
            "deterministic_demo",
            "ollama",
            "openai_compatible",
        }:
            raise ConfigurationError(
                "LLM_PROVIDER must be deterministic_demo, ollama, or openai_compatible"
            )
        if self.llm_provider in {"ollama", "openai_compatible"}:
            missing = [
                name
                for name, value in {
                    "LLM_MODEL": self.llm_model,
                    "LLM_BASE_URL": self.llm_base_url,
                }.items()
                if not value
            ]
            if missing:
                raise ConfigurationError(f"Missing configuration: {', '.join(missing)}")
        if self.retrieval_backend not in {"lexical_demo", "semantic", "hybrid_ollama"}:
            raise ConfigurationError(
                "RETRIEVAL_BACKEND must be lexical_demo, semantic, or hybrid_ollama"
            )
        if self.llm_structured_output_mode not in {"json_schema", "json_object"}:
            raise ConfigurationError(
                "LLM_STRUCTURED_OUTPUT_MODE must be json_schema or json_object"
            )
        if self.llm_reasoning_effort not in {"", "none", "default", "low", "medium", "high"}:
            raise ConfigurationError(
                "LLM_REASONING_EFFORT must be empty, none, default, low, medium, or high"
            )
        if self.session_backend not in {"memory", "upstash_redis"}:
            raise ConfigurationError("SESSION_BACKEND must be memory or upstash_redis")
        if self.session_backend == "upstash_redis":
            missing = [
                name
                for name, value in {
                    "UPSTASH_REDIS_REST_URL": self.upstash_redis_rest_url,
                    "UPSTASH_REDIS_REST_TOKEN": self.upstash_redis_rest_token,
                }.items()
                if not value
            ]
            if missing:
                raise ConfigurationError(f"Missing configuration: {', '.join(missing)}")
