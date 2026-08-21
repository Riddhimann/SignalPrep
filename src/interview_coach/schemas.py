from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NormalizedEmotion = Literal[
    "positive_confident", "neutral", "uncertain_hesitant", "negative_frustrated"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentChunk(StrictModel):
    chunk_id: str
    source: Literal["resume", "job_description", "rubric"]
    section: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedEvidence(StrictModel):
    chunk_id: str
    source: Literal["resume", "job_description", "rubric"]
    text: str
    score: float = Field(ge=0, le=1)
    retrieval_method: str = "lexical"
    lexical_score: float | None = Field(default=None, ge=0, le=1)
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    risk_flags: list[str] = Field(default_factory=list)


class GroundingAudit(StrictModel):
    status: Literal["grounded", "partial", "ungrounded", "no_evidence"]
    citation_valid: bool
    support_score: float = Field(ge=0, le=1)
    lexical_support_score: float = Field(default=0, ge=0, le=1)
    semantic_support_score: float | None = Field(default=None, ge=0, le=1)
    support_method: Literal["lexical", "semantic_indicator", "none"] = "none"
    cited_evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class RuntimeProvenance(StrictModel):
    generation_provider: str
    model_name: str
    structured_output_mode: str
    retrieval_configured: str
    retrieval_effective: str
    embedding_model: str
    prompt_version: str
    rubric_version: str


class EmotionPrediction(StrictModel):
    raw_label: str
    normalized_label: NormalizedEmotion
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_id: str

    @model_validator(mode="after")
    def validate_probabilities(self) -> EmotionPrediction:
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        return self


class CommunicationSignal(StrictModel):
    label: NormalizedEmotion | Literal["unavailable"]
    confidence: float = Field(ge=0, le=1)
    status: Literal["available", "low_confidence", "unavailable"]
    explanation: str
    modality: Literal["multimodal", "speech_only", "text_only", "none"] = "none"
    probabilities: dict[str, float] = Field(default_factory=dict)


class Scorecard(StrictModel):
    relevance: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    structure: int = Field(ge=1, le=10)
    technical_depth: int = Field(ge=1, le=10)
    evidence: int = Field(ge=1, le=10)


class ScoreCalibration(StrictModel):
    method: Literal["hybrid_anchored_v1"]
    model_weight: float = Field(ge=0, le=1)
    raw_model_scores: Scorecard
    observable_scores: Scorecard
    signals: dict[str, float]


class AnswerEvaluation(StrictModel):
    scores: Scorecard
    strengths: list[str] = Field(max_length=3)
    improvements: list[str] = Field(max_length=3)
    improved_answer_outline: list[str]
    evidence_used: list[str]
    next_action_suggestion: Literal["probe", "clarify", "change_topic", "finish"]
    suggested_next_question: str
    calibration: ScoreCalibration | None = None


class RequirementExtraction(StrictModel):
    target_role: str
    responsibilities: list[str]
    required_skills: list[str]
    preferred_skills: list[str]
    evaluation_topics: list[str]


class GeneratedQuestion(StrictModel):
    question: str = Field(min_length=5)
    topic: str
    evidence_ids: list[str]
    rationale: str


class TranscriptResult(StrictModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    latency_ms: float
    model_id: str


class InterviewTurn(StrictModel):
    turn_number: int = Field(ge=1)
    question: str
    topic: str
    transcript: str
    speech_emotion: EmotionPrediction | None = None
    text_emotion: EmotionPrediction | None = None
    communication_signal: CommunicationSignal
    evidence: list[RetrievedEvidence]
    question_evidence_ids: list[str] = Field(default_factory=list)
    question_grounding: GroundingAudit | None = None
    evaluation: AnswerEvaluation
    controller_action: Literal["probe", "clarify", "change_topic", "finish"]
    next_question: str | None = None


class InterviewSession(StrictModel):
    session_id: str
    role: str
    interview_type: Literal["behavioral", "technical", "mixed"]
    difficulty: Literal["beginner", "intermediate", "advanced"]
    max_questions: int = Field(default=5, ge=1, le=10)
    required_skills: list[str] = Field(default_factory=list)
    covered_skills: list[str] = Field(default_factory=list)
    current_question: str | None = None
    current_topic: str | None = None
    current_question_evidence_ids: list[str] = Field(default_factory=list)
    current_question_grounding: GroundingAudit | None = None
    turns: list[InterviewTurn] = Field(default_factory=list)
    status: Literal["created", "ready", "active", "completed", "failed"] = "created"
    degraded_modes: list[str] = Field(default_factory=list)
    security_events: list[str] = Field(default_factory=list)
    runtime: RuntimeProvenance | None = None


class TurnResult(StrictModel):
    turn: InterviewTurn
    session_status: str


class FinalReport(StrictModel):
    session_id: str
    role: str
    completed_questions: int
    score_trends: dict[str, list[int]]
    recurring_strengths: list[str]
    recurring_improvements: list[str]
    practice_plan: list[str]
    disclaimer: str
    markdown: str


class ModelStatus(StrictModel):
    component: str
    status: Literal["available", "degraded", "unavailable"]
    detail: str
