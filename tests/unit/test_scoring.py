from interview_coach.interview.scoring import calibrate_evaluation, observable_scorecard
from interview_coach.schemas import AnswerEvaluation, Scorecard


def _model_evaluation() -> AnswerEvaluation:
    return AnswerEvaluation(
        scores=Scorecard(
            relevance=8,
            clarity=8,
            structure=8,
            technical_depth=8,
            evidence=8,
        ),
        strengths=["Structured response"],
        improvements=["Add detail"],
        improved_answer_outline=["Context", "Action", "Result"],
        evidence_used=["resume_001"],
        next_action_suggestion="probe",
        suggested_next_question="How did you validate it?",
    )


def test_observable_scoring_separates_specific_and_generic_answers():
    question = "Tell me about a classification model you validated and deployed."
    strong = (
        "I developed a churn classifier using SQL features. I used time-based validation, "
        "calibrated probabilities, compared ROC-AUC and F1, deployed it with FastAPI, and "
        "improved successful outreach by 12 percent."
    )
    weak = "I made a machine learning model in Python. It worked well and then we used it."

    strong_scores, _ = observable_scorecard(question, strong)
    weak_scores, _ = observable_scorecard(question, weak)

    assert sum(strong_scores.model_dump().values()) > sum(weak_scores.model_dump().values())
    assert strong_scores.evidence > weak_scores.evidence
    assert strong_scores.technical_depth > weak_scores.technical_depth


def test_calibration_caps_short_unsupported_answer_despite_high_model_scores():
    calibrated = calibrate_evaluation(
        _model_evaluation(),
        "How did you validate and deploy the model?",
        "I made a model and it worked well.",
    )

    assert calibrated.calibration is not None
    assert calibrated.scores.technical_depth <= 4
    assert calibrated.scores.evidence <= 3
    assert max(calibrated.scores.model_dump().values()) <= 5
