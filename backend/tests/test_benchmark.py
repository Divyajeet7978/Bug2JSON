import json
from pathlib import Path

import pytest

from training.benchmark import load_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_bundled_benchmark_has_policy_verified_synthetic_cases() -> None:
    rows = load_benchmark(PROJECT_ROOT / "data" / "benchmark.jsonl")

    assert len(rows) == 30
    assert len({row["group_id"] for row in rows}) == len(rows)
    assert all(row["source"] == "synthetic_policy_benchmark" for row in rows)


def test_benchmark_rejects_unverified_labels(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    row = {
        "input": "The navigation menu does not open.",
        "output": {
            "category": "ui",
            "severity": "low",
            "priority": "P4",
            "environment": "unknown",
            "reproducible": "unknown",
            "probable_cause": "unknown",
            "suggested_action": "Inspect the menu click handler.",
        },
        "source": "synthetic_policy_benchmark",
        "group_id": "menu",
        "label_status": "unreviewed",
    }
    path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="policy_verified"):
        load_benchmark(path)


def test_benchmark_rejects_duplicate_reports(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    row = {
        "input": "The navigation menu does not open.",
        "output": {
            "category": "ui",
            "severity": "low",
            "priority": "P4",
            "environment": "unknown",
            "reproducible": "unknown",
            "probable_cause": "unknown",
            "suggested_action": "Inspect the menu click handler.",
        },
        "source": "synthetic_policy_benchmark",
        "group_id": "menu",
        "label_status": "policy_verified",
    }
    duplicate = {**row, "group_id": "menu-paraphrase"}
    path.write_text(
        "\n".join(json.dumps(item) for item in (row, duplicate)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate benchmark report"):
        load_benchmark(path)
