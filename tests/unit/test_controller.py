from interview_coach.interview.controller import choose_action
from interview_coach.schemas import AnswerEvaluation, InterviewSession, Scorecard


def evaluation(relevance=7, depth=7, suggestion="change_topic"):
    return AnswerEvaluation(
        scores=Scorecard(
            relevance=relevance,
            clarity=7,
            structure=7,
            technical_depth=depth,
            evidence=7,
        ),
        strengths=[],
        improvements=[],
        improved_answer_outline=[],
        evidence_used=[],
        next_action_suggestion=suggestion,
        suggested_next_question="Next?",
    )


def session(max_questions=5):
    return InterviewSession(
        session_id="s",
        role="Engineer",
        interview_type="technical",
        difficulty="intermediate",
        max_questions=max_questions,
        current_topic="Python",
    )


def test_short_answer_forces_clarify():
    assert choose_action(session(), evaluation(), "Very short answer") == "clarify"


def test_technical_claim_with_low_depth_forces_probe():
    transcript = (
        "I built a model and deployed the pipeline for our users with some useful results and team "
        "feedback after completing the initial implementation and review."
    )
    assert choose_action(session(), evaluation(depth=5), transcript) == "probe"


def test_limit_forces_finish_before_other_rules():
    assert choose_action(session(max_questions=1), evaluation(), "short") == "finish"
