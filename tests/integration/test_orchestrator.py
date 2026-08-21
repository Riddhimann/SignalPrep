import asyncio
from pathlib import Path

from interview_coach.config import Settings
from interview_coach.container import create_container

ROOT = Path(__file__).parents[2]


def test_complete_two_turn_offline_interview():
    settings = Settings(max_interview_questions=2)
    coach = create_container(settings).orchestrator
    resume = (ROOT / "data/samples/sample_resume.txt").read_text(encoding="utf-8")
    jd = (ROOT / "data/samples/sample_jd.txt").read_text(encoding="utf-8")

    session = coach.create_session("Data Scientist", max_questions=2)
    asyncio.run(coach.index_documents(session.session_id, resume, jd))
    started = asyncio.run(coach.start(session.session_id))
    assert started.current_question

    first = asyncio.run(
        coach.process_answer(
            session.session_id,
            "I built a Python model pipeline for customer churn. I used time based validation, "
            "tested calibration, deployed a FastAPI service, and improved successful outreach by 12 percent.",
        )
    )
    assert first.turn.evaluation.evidence_used
    assert first.turn.next_question

    second = asyncio.run(
        coach.process_answer(
            session.session_id,
            "The main trade-off was recall versus precision. I first agreed the cost metric with stakeholders, "
            "then compared three thresholds and monitored the result for 30 days after deployment.",
        )
    )
    assert second.session_status == "completed"
    report = coach.report(session.session_id)
    assert report.completed_questions == 2
    assert "No hiring recommendation" in report.disclaimer
