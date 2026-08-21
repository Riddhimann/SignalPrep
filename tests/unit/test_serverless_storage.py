from __future__ import annotations

import json

from interview_coach.rag.retriever import RagService
from interview_coach.schemas import InterviewSession
from interview_coach.storage.rag_source_store import UpstashRagSourceStore
from interview_coach.storage.session_store import UpstashSessionStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        assert ttl_seconds > 0
        self.values[key] = value

    def get(self, key: str):
        return self.values.get(key)

    def exists(self, key: str) -> bool:
        return key in self.values


def test_upstash_session_store_round_trip_without_shared_process_memory():
    redis = FakeRedis()
    first = UpstashSessionStore("https://example.test", "secret", 3600)
    second = UpstashSessionStore("https://example.test", "secret", 3600)
    first._redis = redis
    second._redis = redis

    session = InterviewSession(
        session_id="session-1",
        role="Data Scientist",
        interview_type="mixed",
        difficulty="intermediate",
    )
    first.create(session)

    restored = second.get("session-1")
    assert restored == session
    assert restored is not session


def test_rag_index_is_rebuilt_from_persisted_sources_on_a_fresh_instance():
    redis = FakeRedis()
    first_sources = UpstashRagSourceStore("https://example.test", "secret", 3600)
    second_sources = UpstashRagSourceStore("https://example.test", "secret", 3600)
    first_sources._redis = redis
    second_sources._redis = redis
    first = RagService("lexical_demo", "", source_store=first_sources)
    second = RagService("lexical_demo", "", source_store=second_sources)

    first.index_session(
        "session-1",
        "Built a Python churn model and deployed a FastAPI service.",
        "The role requires Python, machine learning, and API deployment.",
    )
    assert not second._stores

    results = second.search("session-1", "Python API deployment")
    assert results
    assert second.has_index("session-1")
    assert json.loads(redis.values["signalprep:rag-source:session-1"])["resume"]
