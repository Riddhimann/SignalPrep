from __future__ import annotations

from interview_coach.schemas import CommunicationSignal, EmotionPrediction

LABELS = ("positive_confident", "neutral", "uncertain_hesitant", "negative_frustrated")


def combine_signals(
    speech: EmotionPrediction | None,
    text: EmotionPrediction | None,
    *,
    speech_weight: float = 0.60,
    confidence_threshold: float = 0.55,
) -> CommunicationSignal:
    if speech is None and text is None:
        return CommunicationSignal(
            label="unavailable",
            confidence=0,
            status="unavailable",
            modality="none",
            explanation="Communication-cue models were unavailable; content feedback continued independently.",
        )
    if speech is not None and text is not None:
        text_weight = 1 - speech_weight
        probabilities = {
            label: speech_weight * speech.probabilities.get(label, 0)
            + text_weight * text.probabilities.get(label, 0)
            for label in LABELS
        }
        modality = "multimodal"
        explanation = "Possible cue from a configurable 60% speech / 40% text heuristic; not an internal-state inference."
    else:
        prediction = speech or text
        assert prediction is not None
        probabilities = {label: prediction.probabilities.get(label, 0) for label in LABELS}
        modality = "speech_only" if speech else "text_only"
        explanation = (
            f"Possible cue from {modality.replace('_', ' ')}; the other modality was unavailable."
        )
    label, confidence = max(probabilities.items(), key=lambda item: item[1])
    status = "available" if confidence >= confidence_threshold else "low_confidence"
    if status == "low_confidence":
        explanation += " Confidence is below the configured threshold; treat this cue cautiously."
    return CommunicationSignal(
        label=label,
        confidence=confidence,
        status=status,
        explanation=explanation,
        modality=modality,
        probabilities=probabilities,
    )
