from interview_coach.interview.requirements import ground_requirement_extraction
from interview_coach.schemas import RequirementExtraction


def test_requirement_grounding_removes_invented_skills_and_keeps_supported_items():
    extraction = RequirementExtraction(
        target_role="Data Scientist",
        responsibilities=[
            "Build machine learning models",
            "Manage a global sales team",
        ],
        required_skills=["Python", "Quantum computing"],
        preferred_skills=["FastAPI", "Kubernetes"],
        evaluation_topics=["SQL data extraction", "Blockchain strategy"],
    )
    jd = (
        "Build and validate machine learning models. Use SQL for data extraction. "
        "Required skills: Python and SQL. "
        "Preferred: FastAPI and Docker."
    )

    grounded, removed = ground_requirement_extraction(extraction, jd)

    assert grounded.required_skills == ["Python"]
    assert grounded.preferred_skills == ["FastAPI"]
    assert grounded.responsibilities == ["Build machine learning models"]
    assert grounded.evaluation_topics == ["SQL data extraction"]
    assert any("Quantum computing" in item for item in removed)
    assert any("Blockchain strategy" in item for item in removed)
