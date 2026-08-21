from fastapi.testclient import TestClient

from interview_coach.api.app import create_app
from interview_coach.config import Settings


def test_api_happy_path():
    client = TestClient(create_app(Settings(max_interview_questions=1)))
    created = client.post(
        "/sessions",
        json={
            "role": "Data Scientist",
            "interview_type": "mixed",
            "difficulty": "intermediate",
            "max_questions": 1,
        },
    )
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    indexed = client.post(
        f"/sessions/{session_id}/documents",
        json={
            "resume_text": "Python SQL machine learning project with measured results",
            "job_description": "Requires Python SQL machine learning and communication",
        },
    )
    assert indexed.status_code == 200
    assert client.post(f"/sessions/{session_id}/start").status_code == 200
    result = client.post(
        f"/sessions/{session_id}/answers",
        json={
            "transcript": "I built a Python machine learning model, validated it with a test set, "
            "and improved processing time by 20 percent for 100 users."
        },
    )
    assert result.status_code == 200
    assert result.json()["session_status"] == "completed"
    report = client.get(f"/sessions/{session_id}/report?format=markdown")
    assert report.status_code == 200
    assert "Interview coaching report" in report.text


def test_api_invalid_manual_json_is_a_422():
    client = TestClient(create_app(Settings()))
    response = client.post("/sessions", json={"role": "Data Scientist"})
    session_id = response.json()["session_id"]
    invalid = client.post(
        f"/sessions/{session_id}/documents",
        json={"resume_text": "", "job_description": "Requires Python"},
    )
    assert invalid.status_code == 422


def test_production_frontend_is_served():
    client = TestClient(create_app(Settings()))
    response = client.get("/")
    assert response.status_code == 200
    assert "SignalPrep" in response.text


def test_health_labels_demo_runtime_as_degraded():
    client = TestClient(create_app(Settings()))
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["component_status"]["llm"] == "degraded"
    assert payload["limitations"]
