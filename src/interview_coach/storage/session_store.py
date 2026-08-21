from __future__ import annotations

from threading import RLock
from typing import Protocol

from interview_coach.exceptions import SessionNotFoundError
from interview_coach.schemas import InterviewSession
from interview_coach.storage.upstash_rest import UpstashRestClient


class SessionStore(Protocol):
    def create(self, session: InterviewSession) -> InterviewSession: ...

    def get(self, session_id: str) -> InterviewSession: ...

    def save(self, session: InterviewSession) -> InterviewSession: ...

    def clear(self) -> None: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, InterviewSession] = {}
        self._lock = RLock()

    def create(self, session: InterviewSession) -> InterviewSession:
        with self._lock:
            self._sessions[session.session_id] = session.model_copy(deep=True)
        return session.model_copy(deep=True)

    def get(self, session_id: str) -> InterviewSession:
        with self._lock:
            try:
                return self._sessions[session_id].model_copy(deep=True)
            except KeyError as exc:
                raise SessionNotFoundError(f"Session {session_id!r} was not found") from exc

    def save(self, session: InterviewSession) -> InterviewSession:
        with self._lock:
            if session.session_id not in self._sessions:
                raise SessionNotFoundError(f"Session {session.session_id!r} was not found")
            self._sessions[session.session_id] = session.model_copy(deep=True)
        return session.model_copy(deep=True)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


class UpstashSessionStore:
    """Serverless-safe session persistence with an explicit retention limit."""

    def __init__(self, url: str, token: str, ttl_seconds: int) -> None:
        self._redis = UpstashRestClient(url=url, token=token)
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"signalprep:session:{session_id}"

    def _write(self, session: InterviewSession) -> InterviewSession:
        snapshot = session.model_copy(deep=True)
        self._redis.set(
            self._key(session.session_id), snapshot.model_dump_json(), self._ttl_seconds
        )
        return snapshot

    def create(self, session: InterviewSession) -> InterviewSession:
        return self._write(session)

    def get(self, session_id: str) -> InterviewSession:
        value = self._redis.get(self._key(session_id))
        if value is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found")
        if isinstance(value, str):
            return InterviewSession.model_validate_json(value)
        return InterviewSession.model_validate(value)

    def save(self, session: InterviewSession) -> InterviewSession:
        if not self._redis.exists(self._key(session.session_id)):
            raise SessionNotFoundError(f"Session {session.session_id!r} was not found")
        return self._write(session)

    def clear(self) -> None:
        # Production data is expired by TTL. Broad key deletion is intentionally unsupported.
        return None
