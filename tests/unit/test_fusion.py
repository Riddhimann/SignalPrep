from interview_coach.nlp.fusion import combine_signals
from interview_coach.schemas import EmotionPrediction


def prediction(label, probabilities):
    return EmotionPrediction(
        raw_label=label,
        normalized_label=label,
        confidence=max(probabilities.values()),
        probabilities=probabilities,
        model_id="test-model",
    )


def test_multimodal_fusion_uses_configured_weights():
    speech = prediction("positive_confident", {"positive_confident": 0.8, "neutral": 0.2})
    text = prediction("neutral", {"positive_confident": 0.25, "neutral": 0.75})
    result = combine_signals(speech, text)
    assert result.modality == "multimodal"
    assert result.probabilities["positive_confident"] == 0.58


def test_both_fail_is_explicitly_unavailable():
    result = combine_signals(None, None)
    assert result.status == "unavailable"
    assert result.label == "unavailable"
