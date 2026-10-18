import pytest
from pydantic import ValidationError

from app.inference import ModelOutputError, parse_analysis


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
