import pytest
from pydantic import ValidationError

from app.schemas import AnalyzeRequest, BugAnalysis


def valid_analysis() -> dict[str, object]:
    return {
        "category": "authentication",
        "severity": "medium",
        "priority": "P2",
        "environment": "Chrome",
        "reproducible": True,
        "probable_cause": "The loading state may not reset after login fails",
        "suggested_action": "Handle failed login responses and reset the loading state",
    }


def test_accepts_unknown_predictions() -> None:
    result = BugAnalysis.model_validate(
        {
            **valid_analysis(),
            "severity": "unknown",
            "priority": "unknown",
            "environment": "unknown",
            "reproducible": "unknown",
            "probable_cause": "unknown",
        }
    )
    assert result.reproducible == "unknown"


def test_rejects_unknown_schema_values_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        BugAnalysis.model_validate({**valid_analysis(), "severity": "urgent"})
    with pytest.raises(ValidationError):
        BugAnalysis.model_validate({**valid_analysis(), "verified": True})


def test_request_strips_report_and_enforces_bounds() -> None:
    assert AnalyzeRequest(report="  A sufficiently long report  ").report == (
        "A sufficiently long report"
    )
    with pytest.raises(ValidationError):
        AnalyzeRequest(report="short")
