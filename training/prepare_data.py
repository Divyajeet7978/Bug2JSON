from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.schemas import BugAnalysis

DEFAULT_INPUT = PROJECT_ROOT / "data" / "examples.jsonl"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MIN_EXAMPLES = 30


def normalize_report(report: str) -> str:
    return re.sub(r"\s+", " ", report.casefold()).strip()


def load_reviewed(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_reports: dict[str, str] = {}
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc

        if not isinstance(row, dict) or not isinstance(row.get("input"), str):
            raise ValueError(f"{path}:{line_number}: expected a string 'input'.")
        if not isinstance(row.get("output"), dict):
            raise ValueError(f"{path}:{line_number}: expected an 'output' object.")
        try:
            analysis = BugAnalysis.model_validate(row["output"])
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid output labels: {exc}") from exc

        report = row["input"].strip()
        if not 10 <= len(report) <= 4000:
            raise ValueError(
                f"{path}:{line_number}: input report must contain 10 to 4000 characters."
            )
        if row.get("reviewed") is not True:
            continue
        group_id_value = row.get("group_id")
        if not isinstance(group_id_value, str):
            raise ValueError(f"{path}:{line_number}: reviewed rows need a string group_id.")
        group_id = group_id_value.strip()
        if not group_id:
            raise ValueError(f"{path}:{line_number}: reviewed rows need a stable group_id.")
        source = row.get("source", "unspecified")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}:{line_number}: source must be a non-empty string.")

        fingerprint = hashlib.sha256(normalize_report(report).encode("utf-8")).hexdigest()
        prior_group = seen_reports.get(fingerprint)
        if prior_group is not None and prior_group != group_id:
            raise ValueError(
                f"{path}:{line_number}: duplicate report appears under different group_ids."
            )
        if fingerprint in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate report.")
        seen_ids.add(fingerprint)
        seen_reports[fingerprint] = group_id
        examples.append(
            {
                "input": report,
                "output": analysis.model_dump(),
                "group_id": group_id,
                "source": source.strip(),
                "reviewed": True,
            }
        )
    if len(examples) < MIN_EXAMPLES:
        raise ValueError(
            f"Need at least {MIN_EXAMPLES} human-reviewed examples; found {len(examples)}. "
            "The bundled examples are illustrative and marked unreviewed."
        )
    return examples


def split_examples(
    examples: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(example["group_id"], []).append(example)

    group_ids = list(groups)
    if len(group_ids) < 3:
        raise ValueError(
            "At least three distinct group_id values are required to create "
            "non-empty train, validation, and test splits."
        )
    random.Random(seed).shuffle(group_ids)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    targets = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assigned = {name: 0 for name in buckets}
    for group_id in group_ids:
        name = min(
            buckets,
            key=lambda key: assigned[key] / max(targets[key], 0.01),
        )
        buckets[name].extend(groups[group_id])
        assigned[name] += len(groups[group_id])
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate reviewed bug reports and create group-safe train/validation/test splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_reviewed(args.input)
    splits = split_examples(examples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        output_path = args.output_dir / f"{name}.jsonl"
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{name}: {len(rows)} examples -> {output_path}")
    print(f"Total reviewed examples: {len(examples)}")


if __name__ == "__main__":
    main()
