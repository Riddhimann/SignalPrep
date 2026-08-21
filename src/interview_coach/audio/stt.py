from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from interview_coach.exceptions import ModelUnavailableError
from interview_coach.schemas import TranscriptResult


class SpeechToText(Protocol):
    model_id: str

    def transcribe(self, audio_path: Path, duration_seconds: float) -> TranscriptResult: ...


class UnavailableSpeechToText:
    model_id = "unavailable"

    def transcribe(self, audio_path: Path, duration_seconds: float) -> TranscriptResult:
        raise ModelUnavailableError(
            "Speech-to-text is unavailable. Install the audio extra; typed transcripts still work."
        )


class FasterWhisperAdapter:
    def __init__(self, model_size: str) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ModelUnavailableError("faster-whisper is not installed") from exc
        self.model_id = f"faster-whisper:{model_size}"
        self._model_class = WhisperModel
        self._model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = self._model_class(self._model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: Path, duration_seconds: float) -> TranscriptResult:
        started = time.perf_counter()
        segments, info = self._load().transcribe(str(audio_path), beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise ModelUnavailableError("Speech-to-text returned an empty transcript")
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=duration_seconds,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            model_id=self.model_id,
        )


def create_stt(model_size: str) -> SpeechToText:
    try:
        return FasterWhisperAdapter(model_size)
    except ModelUnavailableError:
        return UnavailableSpeechToText()
