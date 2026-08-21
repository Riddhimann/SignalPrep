from __future__ import annotations

from interview_coach.schemas import NormalizedEmotion

# Exact, reviewable mappings. Unknown labels are rejected instead of guessed.
LABEL_MAP: dict[str, NormalizedEmotion] = {
    "joy": "positive_confident",
    "happy": "positive_confident",
    "happiness": "positive_confident",
    "positive": "positive_confident",
    "excited": "positive_confident",
    "calm": "neutral",
    "neutral": "neutral",
    "surprise": "uncertain_hesitant",
    "surprised": "uncertain_hesitant",
    "fear": "uncertain_hesitant",
    "fearful": "uncertain_hesitant",
    "sad": "negative_frustrated",
    "sadness": "negative_frustrated",
    "angry": "negative_frustrated",
    "anger": "negative_frustrated",
    "disgust": "negative_frustrated",
    "negative": "negative_frustrated",
}


def normalize_label(raw_label: str) -> NormalizedEmotion:
    key = raw_label.strip().casefold()
    if key not in LABEL_MAP:
        raise KeyError(f"No explicit normalization mapping for model label {raw_label!r}")
    return LABEL_MAP[key]


def normalize_distribution(scores: list[dict[str, float | str]]) -> dict[str, float]:
    result = {
        "positive_confident": 0.0,
        "neutral": 0.0,
        "uncertain_hesitant": 0.0,
        "negative_frustrated": 0.0,
    }
    mapped = 0
    for item in scores:
        try:
            label = normalize_label(str(item["label"]))
        except KeyError:
            continue
        result[label] += float(item["score"])
        mapped += 1
    if not mapped:
        raise KeyError("The model returned no labels with explicit mappings")
    total = sum(result.values()) or 1
    return {label: value / total for label, value in result.items()}
