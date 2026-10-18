import pytest
from pydantic import ValidationError

from app.inference import ModelOutputError, ground_analysis, parse_analysis


VALID = (
    '{"category":"ui","severity":"low","priority":"P4","environment":"unknown",'
    '"reproducible":"unknown","probable_cause":"unknown",'
    '"suggested_action":"Collect a browser console log"}'
)


def test_parses_json_embedded_in_model_text() -> None:
    result = parse_analysis(f"Result:\n{VALID}\nDone.")
    assert result.category == "ui"
    assert result.reproducible == "unknown"


def test_rejects_invalid_or_schema_incompatible_model_output() -> None:
    with pytest.raises(ModelOutputError):
        parse_analysis("No JSON here.")
    invalid = VALID.replace('"severity":"low"', '"severity":"urgent"')
    with pytest.raises(ModelOutputError):
        parse_analysis(invalid)


def test_grounding_removes_unsupported_environment_and_repeatability() -> None:
    analysis = parse_analysis(
        VALID.replace('"environment":"unknown"', '"environment":"iOS"')
        .replace('"reproducible":"unknown"', '"reproducible":false')
        .replace('"severity":"low"', '"severity":"critical"')
        .replace('"priority":"P4"', '"priority":"P1"')
    )

    result = ground_analysis(analysis, "The app crashes after clicking Save on an empty form.")

    assert result.environment == "unknown"
    assert result.reproducible == "unknown"
    assert result.severity == "unknown"
    assert result.priority == "unknown"


def test_grounding_keeps_reported_login_context_and_workaround() -> None:
    analysis = parse_analysis(
        VALID.replace('"category":"ui"', '"category":"authentication"')
        .replace('"severity":"low"', '"severity":"critical"')
        .replace('"priority":"P4"', '"priority":"P1"')
        .replace('"environment":"unknown"', '"environment":"Chrome"')
        .replace('"reproducible":"unknown"', '"reproducible":false')
    )

    result = ground_analysis(
        analysis,
        "Whenever I submit the login form on Chrome, it spins; refreshing the page fixes it temporarily.",
    )

    assert result.environment == "Chrome"
    assert result.reproducible is True
    assert result.severity == "medium"
    assert result.priority == "P2"


def test_grounding_preserves_critical_severity_with_explicit_outage_evidence() -> None:
    analysis = parse_analysis(
        VALID.replace('"severity":"low"', '"severity":"critical"')
        .replace('"priority":"P4"', '"priority":"P1"')
    )

    result = ground_analysis(analysis, "A widespread outage is causing data loss for all users.")

    assert result.severity == "critical"
    assert result.priority == "P1"


def test_grounding_does_not_treat_denied_outage_or_workaround_as_evidence() -> None:
    analysis = parse_analysis(
        VALID.replace('"severity":"low"', '"severity":"critical"')
        .replace('"priority":"P4"', '"priority":"P1"')
    )

    denied_outage = ground_analysis(
        analysis,
        "The issue is not a widespread outage and refreshing does not fix it.",
    )
    denied_workaround = ground_analysis(
        analysis,
        "The application crashes but there is no workaround.",
    )

    assert denied_outage.severity == "unknown"
    assert denied_outage.priority == "unknown"
    assert denied_workaround.severity == "unknown"
    assert denied_workaround.priority == "unknown"


def test_grounding_is_conservative_for_conflicting_report_evidence() -> None:
    analysis = parse_analysis(
        VALID.replace('"reproducible":"unknown"', '"reproducible":false')
        .replace('"severity":"low"', '"severity":"critical"')
        .replace('"priority":"P4"', '"priority":"P1"')
    )

    conflicting_reproducibility = ground_analysis(
        analysis,
        "It happens whenever, but QA cannot reproduce the issue consistently.",
    )
    single_user_outage = ground_analysis(
        analysis,
        "A single-user outage affects the report page.",
    )

    assert conflicting_reproducibility.reproducible == "unknown"
    assert single_user_outage.severity == "unknown"
    assert single_user_outage.priority == "unknown"
