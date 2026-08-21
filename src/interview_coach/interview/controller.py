from __future__ import annotations

import re
from typing import Literal

from interview_coach.schemas import AnswerEvaluation, InterviewSession

ControllerAction = Literal["probe", "clarify", "change_topic", "finish"]
TECHNICAL_CLAIM = re.compile(
    r"\b(model|algorithm|pipeline|api|database|accuracy|latency|deployed|trained|optimized|implemented|built)\b",
    re.IGNORECASE,
)


def contains_technical_claim(transcript: str) -> bool:
    return bool(TECHNICAL_CLAIM.search(transcript))


def choose_action(
    session: InterviewSession, evaluation: AnswerEvaluation, transcript: str
) -> ControllerAction:
    # This answer becomes the next completed turn.
    if len(session.turns) + 1 >= session.max_questions:
        return "finish"
    if len(transcript.split()) < 20 or evaluation.scores.relevance <= 4:
        return "clarify"
    if evaluation.scores.technical_depth <= 6 and contains_technical_claim(transcript):
        return "probe"
    topic_count = sum(turn.topic == session.current_topic for turn in session.turns) + 1
    if topic_count >= 2:
        return "change_topic"
    return evaluation.next_action_suggestion
