import json
from pathlib import Path

import pytest

from training.prepare_data import load_reviewed, split_examples


def _row(report: str, group: str, reviewed: bool = True) -> dict:
    return {
        "input": report,
        "output": {
            "category": "ui",
            "severity": "unknown",
            "priority": "unknown",
            "environment": "unknown",
            "reproducible": "unknown",
            "probable_cause": "unknown",
            "suggested_action": "Collect logs",
        },
        "group_id": group,
        "reviewed": reviewed,
    }


def test_unreviewed_rows_are_excluded(tmp_path: Path) -> None:
    path = tmp_path / "reports.jsonl"
    examples = [
        _row(f"Report number {index}", f"issue-{index}")
        for index in range(30)
    ]
    examples.append(_row("Illustrative only report", "synthetic", reviewed=False))
    path.write_text(
        "".join(json.dumps(example) + "\n" for example in examples),
        encoding="utf-8",
    )

    loaded = load_reviewed(path)
    assert len(loaded) == 30
    assert all(row["reviewed"] is True for row in loaded)
    assert all(row["source"] == "unspecified" for row in loaded)


def test_split_keeps_related_group_together() -> None:
    examples = [
        {"input": f"report {index}", "output": {}, "group_id": f"group-{index // 2}"}
        for index in range(30)
    ]
    splits = split_examples(examples, seed=42)
    membership = {
        row["group_id"]: name
        for name, rows in splits.items()
        for row in rows
    }
    assert len(membership) == 15
    assert sum(map(len, splits.values())) == len(examples)
    assert all(splits[name] for name in ("train", "validation", "test"))


def test_split_requires_three_independent_groups() -> None:
    examples = [
        {"input": f"report {index}", "output": {}, "group_id": f"group-{index // 15}"}
        for index in range(30)
    ]
    with pytest.raises(ValueError, match="At least three distinct group_id"):
        split_examples(examples, seed=42)


def test_reviewed_flag_must_be_boolean_true(tmp_path: Path) -> None:
    path = tmp_path / "reports.jsonl"
    examples = [
        _row(f"Report number {index}", f"issue-{index}")
        for index in range(31)
    ]
    examples[0]["reviewed"] = "false"
    path.write_text(
        "".join(json.dumps(example) + "\n" for example in examples),
        encoding="utf-8",
    )

    assert len(load_reviewed(path)) == 30
