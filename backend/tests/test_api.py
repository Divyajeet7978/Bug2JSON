from fastapi.testclient import TestClient

from app import main
from app.schemas import BugAnalysis


def valid_analysis() -> BugAnalysis:
    return BugAnalysis(
        category="authentication",
        severity="medium",
        priority="P2",
        environment="Chrome",
        reproducible=True,
        probable_cause="The loading state may not reset after login fails",
        suggested_action="Handle failed login responses and reset the loading state",
    )


def test_status_reports_configured_model() -> None:
    response = TestClient(main.app).get("/api/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["adapter_available"], bool)


def test_analyze_returns_validated_result(monkeypatch) -> None:
    monkeypatch.setattr(
        main.analyzer,
        "analyze",
        lambda report, mode: (valid_analysis(), "test-model", 23),
    )

    response = TestClient(main.app).post(
        "/api/analyze",
        json={"report": "The login page keeps spinning on Chrome."},
    )

    assert response.status_code == 200
    assert response.json()["analysis"]["category"] == "authentication"
    assert response.json()["mode"] == "base"
    assert response.json()["elapsed_ms"] == 23


def test_analyze_surfaces_missing_model_as_service_unavailable(monkeypatch) -> None:
    def unavailable(report, mode):
        raise main.ModelUnavailableError("Model dependencies are missing.")

    monkeypatch.setattr(main.analyzer, "analyze", unavailable)
    response = TestClient(main.app).post(
        "/api/analyze",
        json={"report": "The login page keeps spinning on Chrome.", "mode": "base"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model dependencies are missing."


def test_analyze_rejects_short_reports() -> None:
    response = TestClient(main.app).post(
        "/api/analyze",
        json={"report": "short"},
    )

    assert response.status_code == 422
