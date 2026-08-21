from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from interview_coach.exceptions import AudioError


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    path: Path
    duration_seconds: float
    temporary: bool = False

    def cleanup(self) -> None:
        if self.temporary:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


def _wav_metadata(path: Path) -> tuple[int, int, int, float]:
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            channels = audio.getnchannels()
            width = audio.getsampwidth()
            duration = audio.getnframes() / rate if rate else 0
        return rate, channels, width, duration
    except wave.Error as exc:
        raise AudioError("The WAV file is corrupt or unsupported") from exc


def preprocess_audio(audio_path: str | Path, *, max_bytes: int, max_seconds: int) -> PreparedAudio:
    path = Path(audio_path)
    if not path.is_file():
        raise AudioError("Audio file does not exist")
    size = path.stat().st_size
    if size <= 44:
        raise AudioError("Audio file is empty")
    if size > max_bytes:
        raise AudioError(f"Audio exceeds the {max_bytes}-byte limit")

    if path.suffix.lower() == ".wav":
        rate, channels, width, duration = _wav_metadata(path)
        if duration <= 0 or duration > max_seconds:
            raise AudioError(f"Audio duration must be between 0 and {max_seconds} seconds")
        if rate == 16000 and channels == 1 and width == 2:
            return PreparedAudio(path=path, duration_seconds=duration)

    try:
        import librosa  # type: ignore
        import soundfile as sf  # type: ignore
    except ImportError as exc:
        raise AudioError(
            "Non-mono-16kHz audio conversion requires the 'audio' dependency extra"
        ) from exc
    try:
        waveform, _ = librosa.load(str(path), sr=16000, mono=True, duration=max_seconds + 1)
    except Exception as exc:
        raise AudioError("Could not decode the audio file") from exc
    duration = len(waveform) / 16000
    if duration <= 0 or duration > max_seconds:
        raise AudioError(f"Audio duration must be between 0 and {max_seconds} seconds")
    handle, output_name = tempfile.mkstemp(suffix=".wav", prefix="interview-coach-")
    os.close(handle)
    sf.write(output_name, waveform, 16000, subtype="PCM_16")
    return PreparedAudio(path=Path(output_name), duration_seconds=duration, temporary=True)
