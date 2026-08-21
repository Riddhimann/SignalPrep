from __future__ import annotations

from typing import Any

import httpx


class UpstashRestClient:
    """Small Redis REST client for the GET/SET/EXISTS operations SignalPrep needs."""

    def __init__(self, url: str, token: str, timeout_seconds: float = 10) -> None:
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout_seconds

    def execute(self, *command: str | int) -> Any:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url,
                headers=self._headers,
                json=list(command),
            )
            response.raise_for_status()
            payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Redis command failed: {payload['error']}")
        return payload.get("result")

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.execute("SET", key, value, "EX", ttl_seconds)

    def get(self, key: str) -> Any:
        return self.execute("GET", key)

    def exists(self, key: str) -> bool:
        return bool(self.execute("EXISTS", key))
