from __future__ import annotations

import re

from interview_coach.schemas import (
    AnswerEvaluation,
    ScoreCalibration,
    Scorecard,
)

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}")
_NUMBER = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|seconds?|hours?|days?|x|users?|records?)?\b",
    re.IGNORECASE,
)
_VALIDATION = {
    "a/b",
    "calibrated",
    "calibration",
    "cross-validation",
    "evaluated",
    "f1",
    "guardrail",
    "metric",
    "monitored",
    "precision",
    "recall",
    "roc-auc",
    "test",
    "validated",
    "validation",
}
_TECHNICAL = {
    "api",
    "calibration",
    "classifier",
    "database",
    "docker",
    "embedding",
    "fastapi",
    "feature",
    "latency",
    "model",
    "pipeline",
    "python",
    "rag",
    "retrieval",
    "schema",
    "sql",
    "threshold",
}
_OUTCOME = {
    "achieved",
    "improved",
    "increased",
    "reduced",
    "result",
    "outcome",
    "used the result",
}
_SEQUENCE = {"first", "then", "after", "before", "finally", "next", "so that"}
_DECISION = {
    "because",
    "chose",
    "compared",
    "decision",
    "selected",
    "trade-off",
    "tradeoff",
    "why",
}
_STOPWORDS = {
    "about",
    "and",
    "describe",
    "did",
    "for",
    "from",
    "how",
    "that",
    "the",
    "this",
    "tell",
    "what",
    "when",
    "with",
    "you",
    "your",
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOPWORDS}


def _phrase_hits(text: str, phrases: set[str]) -> int:
    lower = text.lower()
    return sum(phrase in lower for phrase in phrases)


def _bounded(value: float) -> int:
    return max(1, min(10, round(value)))


def observable_scorecard(question: str, transcript: str) -> tuple[Scorecard, dict[str, float]]:
    words = transcript.split()
    word_count = len(words)
    sentence_count = len([item for item in re.split(r"[.!?]+", transcript) if item.strip()])
    question_tokens = _tokens(question)
    answer_tokens = _tokens(transcript)
    overlap_ratio = len(question_tokens & answer_tokens) / max(len(question_tokens), 1)
    validation_hits = _phrase_hits(transcript, _VALIDATION)
    technical_hits = _phrase_hits(transcript, _TECHNICAL)
    outcome_hits = _phrase_hits(transcript, _OUTCOME)
    sequence_hits = _phrase_hits(transcript, _SEQUENCE)
    decision_hits = _phrase_hits(transcript, _DECISION)
    numeric_hits = len(_NUMBER.findall(transcript))

    relevance = 3 + min(2.5, overlap_ratio * 5) + (1.5 if word_count >= 30 else 0)
    clarity = 3 + (2 if 20 <= word_count <= 180 else 0) + min(2, sentence_count / 2)
    clarity += 1 if decision_hits or sequence_hits else 0
    structure = 2 + min(2, sentence_count / 2) + min(1.5, sequence_hits)
    structure += 1.5 if validation_hits else 0
    structure += 1.5 if outcome_hits or numeric_hits else 0
    technical_depth = 2 + min(3, technical_hits / 2) + min(2, validation_hits)
    technical_depth += min(1.5, decision_hits)
    evidence = 1 + min(2.5, numeric_hits * 1.5) + min(2, validation_hits)
    evidence += min(2, outcome_hits) + (1 if word_count >= 35 else 0)

    scores = Scorecard(
        relevance=_bounded(relevance),
        clarity=_bounded(clarity),
        structure=_bounded(structure),
        technical_depth=_bounded(technical_depth),
        evidence=_bounded(evidence),
    )
    signals = {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "question_token_overlap": overlap_ratio,
        "validation_hits": float(validation_hits),
        "technical_hits": float(technical_hits),
        "outcome_hits": float(outcome_hits),
        "sequence_hits": float(sequence_hits),
        "decision_hits": float(decision_hits),
        "numeric_hits": float(numeric_hits),
    }
    return scores, signals


def calibrate_evaluation(
    evaluation: AnswerEvaluation,
    question: str,
    transcript: str,
    *,
    model_weight: float = 0.35,
) -> AnswerEvaluation:
    if model_weight < 0 or model_weight > 1:
        raise ValueError("model_weight must be between 0 and 1")
    observable, signals = observable_scorecard(question, transcript)
    raw = evaluation.scores
    combined: dict[str, int] = {}
    for field in Scorecard.model_fields:
        value = model_weight * getattr(raw, field) + (1 - model_weight) * getattr(observable, field)
        combined[field] = _bounded(value)

    word_count = int(signals["word_count"])
    if word_count < 20:
        combined = {field: min(value, 5) for field, value in combined.items()}
    if signals["validation_hits"] == 0 and signals["technical_hits"] < 2:
        combined["technical_depth"] = min(combined["technical_depth"], 4)
    if signals["numeric_hits"] == 0 and signals["validation_hits"] == 0:
        combined["evidence"] = min(combined["evidence"], 3)

    calibrated = Scorecard(**combined)
    audit = ScoreCalibration(
        method="hybrid_anchored_v1",
        model_weight=model_weight,
        raw_model_scores=raw,
        observable_scores=observable,
        signals=signals,
    )
    return evaluation.model_copy(update={"scores": calibrated, "calibration": audit})
