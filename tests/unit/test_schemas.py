import pytest
from pydantic import ValidationError

from interview_coach.schemas import Scorecard


def test_scorecard_enforces_rubric_range():
    with pytest.raises(ValidationError):
        Scorecard(relevance=11, clarity=8, structure=8, technical_depth=8, evidence=8)


def test_scorecard_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Scorecard(
            relevance=8,
            clarity=8,
            structure=8,
            technical_depth=8,
            evidence=8,
            employability=9,
        )
