import json
from pathlib import Path

import pytest

from training.evaluate import load_test_rows, parse_prediction, score


def _analysis() -> dict[str, object]:
    return {
        "category": "ui",
        "severity": "low",
        "priority": "P4",
        "environment": "unknown",
        "reproducible": "unknown",
        "probable_cause": "unknown",
        "suggested_action": "Collect a browser console log",
    }


def _row(reviewed: bool = True) -> dict[str, object]:
    return {
        "input": "The navigation menu does not open.",
        "output": _analysis(),
        "group_id": "navigation-menu",
        "reviewed": reviewed,
    }


def test_score_separates_valid_json_from_schema_validity() -> None:
    metrics = score(
        [{"output": _analysis()}],
        [(None, True)],
    )

    assert metrics["json_validity"] == 1.0
    assert metrics["schema_validity"] == 0.0
    assert metrics["all_fields_exact_match"] == 0.0


def test_score_requires_one_prediction_per_row() -> None:
    with pytest.raises(ValueError, match="number of predictions"):
        score([{"output": _analysis()}], [])


def test_prediction_json_validity_requires_a_bare_json_document() -> None:
    valid_result, valid_json = parse_prediction(json.dumps(_analysis()))
    fenced_result, fenced_json = parse_prediction(f"```json\n{json.dumps(_analysis())}\n```")
    invalid_schema_result, invalid_schema_json = parse_prediction('{"severity":"urgent"}')
    nonstandard_result, nonstandard_json = parse_prediction('{"value":NaN}')

    assert valid_json and valid_result is not None
    assert not fenced_json and fenced_result is None
    assert invalid_schema_json and invalid_schema_result is None
    assert not nonstandard_json and nonstandard_result is None


def test_test_split_must_contain_reviewed_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(_row(reviewed=False)), encoding="utf-8")

    with pytest.raises(ValueError, match="human-reviewed"):
        load_test_rows(path)
