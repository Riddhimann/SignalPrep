from __future__ import annotations

from collections import Counter

from interview_coach.schemas import FinalReport, InterviewSession

DISCLAIMER = (
    "Practice-only coaching report. Communication cues are uncertain and are not evidence of "
    "personality, honesty, mental health, competence, or employability. No hiring recommendation is made."
)


def build_report(session: InterviewSession) -> FinalReport:
    score_names = ("relevance", "clarity", "structure", "technical_depth", "evidence")
    trends = {
        name: [getattr(turn.evaluation.scores, name) for turn in session.turns]
        for name in score_names
    }
    strength_counts = Counter(item for turn in session.turns for item in turn.evaluation.strengths)
    improvement_counts = Counter(
        item for turn in session.turns for item in turn.evaluation.improvements
    )
    strengths = [item for item, _ in strength_counts.most_common(4)]
    improvements = [item for item, _ in improvement_counts.most_common(4)]
    practice = [f"Practice: {item}" for item in improvements[:3]]
    if not practice:
        practice = ["Practice two timed answers using STAR and review them for specificity."]

    lines = [
        f"# Interview coaching report: {session.role}",
        "",
        f"Questions completed: {len(session.turns)}",
        "",
        "## Score trends",
        "",
    ]
    lines += [f"- {name.replace('_', ' ').title()}: {values}" for name, values in trends.items()]
    lines += ["", "## Recurring strengths", ""]
    lines += [f"- {item}" for item in strengths] or ["- Complete more turns to identify a trend."]
    lines += ["", "## Recurring improvements", ""]
    lines += [f"- {item}" for item in improvements] or ["- No recurring gap detected."]
    lines += ["", "## Practice plan", ""] + [f"- {item}" for item in practice]
    lines += ["", "## Responsible-use note", "", DISCLAIMER]
    return FinalReport(
        session_id=session.session_id,
        role=session.role,
        completed_questions=len(session.turns),
        score_trends=trends,
        recurring_strengths=strengths,
        recurring_improvements=improvements,
        practice_plan=practice,
        disclaimer=DISCLAIMER,
        markdown="\n".join(lines),
    )
