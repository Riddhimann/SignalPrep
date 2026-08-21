from __future__ import annotations

import json
from typing import Protocol

from interview_coach.storage.upstash_rest import UpstashRestClient


class RagSourceStore(Protocol):
    def save(
        self,
        session_id: str,
        resume: str,
        job_description: str,
        effective_mode: str,
    ) -> None: ...

    def load(self, session_id: str) -> tuple[str, str, str] | None: ...


class UpstashRagSourceStore:
    """Persists source text so a fresh serverless instance can rebuild its retrieval index."""

    def __init__(self, url: str, token: str, ttl_seconds: int) -> None:
        self._redis = UpstashRestClient(url=url, token=token)
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"signalprep:rag-source:{session_id}"

    def save(
        self,
        session_id: str,
        resume: str,
        job_description: str,
        effective_mode: str,
    ) -> None:
        value = json.dumps(
            {
                "resume": resume,
                "job_description": job_description,
                "effective_mode": effective_mode,
            }
        )
        self._redis.set(self._key(session_id), value, self._ttl_seconds)

    def load(self, session_id: str) -> tuple[str, str, str] | None:
        value = self._redis.get(self._key(session_id))
        if value is None:
            return None
        payload = json.loads(value) if isinstance(value, str) else value
        return (
            str(payload["resume"]),
            str(payload["job_description"]),
            str(payload["effective_mode"]),
        )
