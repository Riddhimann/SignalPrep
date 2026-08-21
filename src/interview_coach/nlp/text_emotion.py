from __future__ import annotations

from typing import Protocol

from interview_coach.exceptions import ModelUnavailableError
from interview_coach.nlp.label_mapping import normalize_distribution
from interview_coach.schemas import EmotionPrediction


class TextEmotionAnalyzer(Protocol):
    model_id: str

    def predict(self, text: str) -> EmotionPrediction: ...


class UnavailableTextEmotion:
    model_id = "unavailable"

    def predict(self, text: str) -> EmotionPrediction:
        raise ModelUnavailableError("Text communication-cue model is not configured")


class HuggingFaceTextEmotion:
    def __init__(self, model_id: str) -> None:
        if not model_id:
            raise ModelUnavailableError("TEXT_EMOTION_MODEL is not configured")
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ModelUnavailableError("transformers is not installed") from exc
        self.model_id = model_id
        self._pipeline_factory = pipeline
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                "text-classification", model=self.model_id, top_k=None, device=-1
            )
        return self._pipeline

    def predict(self, text: str) -> EmotionPrediction:
        if not text.strip():
            raise ValueError("Cannot analyze empty text")
        output = self._load()(text, truncation=True)
        scores = output[0] if output and isinstance(output[0], list) else output
        probabilities = normalize_distribution(scores)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        raw = max(scores, key=lambda item: item["score"])["label"]
        return EmotionPrediction(
            raw_label=str(raw),
            normalized_label=label,
            confidence=confidence,
            probabilities=probabilities,
            model_id=self.model_id,
        )


def create_text_emotion(model_id: str) -> TextEmotionAnalyzer:
    try:
        return HuggingFaceTextEmotion(model_id)
    except ModelUnavailableError:
        return UnavailableTextEmotion()
