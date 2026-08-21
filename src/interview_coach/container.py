from __future__ import annotations

from dataclasses import dataclass

from interview_coach.audio.speech_emotion import create_speech_emotion
from interview_coach.audio.stt import create_stt
from interview_coach.config import Settings
from interview_coach.interview.orchestrator import InterviewOrchestrator
from interview_coach.llm.gateway import create_gateway
from interview_coach.nlp.text_emotion import create_text_emotion
from interview_coach.rag.retriever import RagService
from interview_coach.storage.rag_source_store import UpstashRagSourceStore
from interview_coach.storage.session_store import InMemorySessionStore, UpstashSessionStore


@dataclass(slots=True)
class Container:
    settings: Settings
    orchestrator: InterviewOrchestrator


def create_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings.from_env()
    if settings.session_backend == "upstash_redis":
        store = UpstashSessionStore(
            settings.upstash_redis_rest_url,
            settings.upstash_redis_rest_token,
            settings.session_ttl_seconds,
        )
        rag_source_store = UpstashRagSourceStore(
            settings.upstash_redis_rest_url,
            settings.upstash_redis_rest_token,
            settings.session_ttl_seconds,
        )
    else:
        store = InMemorySessionStore()
        rag_source_store = None
    orchestrator = InterviewOrchestrator(
        settings=settings,
        store=store,
        rag=RagService(
            settings.retrieval_backend,
            settings.embedding_model,
            ollama_url=settings.llm_base_url or "http://127.0.0.1:11434",
            timeout_seconds=settings.llm_timeout_seconds,
            source_store=rag_source_store,
        ),
        gateway=create_gateway(settings),
        stt=create_stt(settings.stt_model_size),
        speech_emotion=create_speech_emotion(settings.speech_emotion_model),
        text_emotion=create_text_emotion(settings.text_emotion_model),
    )
    return Container(settings=settings, orchestrator=orchestrator)
