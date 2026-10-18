from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import BugAnalysis

from training.evaluate import FIELDS, score

DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "benchmark.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "benchmark-results.json"


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: set[str] = set()
    reports: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict) or row.get("label_status") != "policy_verified":
            raise ValueError(
                f"{path}:{line_number}: benchmark rows need label_status "
                "'policy_verified'."
            )
        if row.get("source") != "synthetic_policy_benchmark":
            raise ValueError(
                f"{path}:{line_number}: benchmark rows must be synthetic policy cases."
            )
        report = row.get("input")
        group_id = row.get("group_id")
        if not isinstance(report, str) or not 10 <= len(report.strip()) <= 4000:
            raise ValueError(
                f"{path}:{line_number}: input must contain 10 to 4000 characters."
            )
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{path}:{line_number}: expected a stable group_id.")
        group_id = group_id.strip()
        if group_id in groups:
            raise ValueError(f"{path}:{line_number}: duplicate group_id {group_id!r}.")
        fingerprint = hashlib.sha256(
            " ".join(report.casefold().split()).encode("utf-8")
        ).hexdigest()
        if fingerprint in reports:
            raise ValueError(f"{path}:{line_number}: duplicate benchmark report.")
        try:
            output = BugAnalysis.model_validate(row.get("output")).model_dump()
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid expected labels: {exc}") from exc
        rows.append(
            {
                "input": report.strip(),
                "output": output,
                "group_id": group_id,
                "source": row["source"],
                "label_status": row["label_status"],
            }
        )
        groups.add(group_id)
        reports.add(fingerprint)
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}.")
    return rows


def _accuracy(
    rows: list[dict[str, Any]],
    predictions: list[tuple[dict[str, Any] | None, bool]],
    field: str,
) -> float:
    return sum(
        prediction is not None and prediction[field] == row["output"][field]
        for row, (prediction, _) in zip(rows, predictions)
    ) / len(rows)


def run_benchmark(
    benchmark_path: Path,
    api_url: str,
    output_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    rows = load_benchmark(benchmark_path)
    root_url = api_url.rstrip("/")
    predictions: list[tuple[dict[str, Any] | None, bool]] = []
    cases: list[dict[str, Any]] = []
    with httpx.Client(
        timeout=httpx.Timeout(timeout_seconds, connect=5.0),
        follow_redirects=False,
    ) as client:
        status_response = client.get(f"{root_url}/api/status")
        status_response.raise_for_status()
        status = status_response.json()
        if not isinstance(status, dict) or not isinstance(status.get("base_model"), str):
            raise ValueError("The API returned an invalid model status response.")
        model_name = status["base_model"]

        for index, row in enumerate(rows, 1):
            prediction: dict[str, Any] | None = None
            error: str | None = None
            elapsed_ms: int | None = None
            try:
                response = client.post(
                    f"{root_url}/api/analyze",
                    json={"report": row["input"], "mode": "base"},
                )
                response.raise_for_status()
                payload = response.json()
                if (
                    not isinstance(payload, dict)
                    or payload.get("mode") != "base"
                    or payload.get("model_name") != model_name
                ):
                    raise ValueError("The API response did not match the base-model run.")
                prediction = BugAnalysis.model_validate(payload.get("analysis")).model_dump()
                elapsed_ms = payload.get("elapsed_ms")
                if not isinstance(elapsed_ms, int) or elapsed_ms < 0:
                    raise ValueError("The API returned invalid inference latency.")
            except (httpx.HTTPError, ValueError, ValidationError) as exc:
                error = str(exc)
            predictions.append((prediction, prediction is not None))
            exact_match = prediction is not None and all(
                prediction[field] == row["output"][field] for field in FIELDS
            )
            cases.append(
                {
                    "group_id": row["group_id"],
                    "input": row["input"],
                    "expected": row["output"],
                    "prediction": prediction,
                    "exact_match": exact_match,
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                }
            )
            print(f"[{index}/{len(rows)}] {row['group_id']}: {'OK' if prediction else 'FAILED'}")

    metrics = score(rows, predictions)
    metrics.pop("json_validity")
    metrics["schema_validity"] = sum(
        prediction is not None for prediction, _ in predictions
    ) / len(rows)
    metrics["category_accuracy"] = _accuracy(rows, predictions, "category")
    metrics["severity_accuracy"] = _accuracy(rows, predictions, "severity")
    metrics["priority_accuracy"] = _accuracy(rows, predictions, "priority")
    metrics["environment_accuracy"] = _accuracy(rows, predictions, "environment")
    metrics["reproducibility_accuracy"] = _accuracy(rows, predictions, "reproducible")
    result = {
        "benchmark": "Bug2JSON synthetic policy benchmark",
        "benchmark_file": str(benchmark_path.relative_to(PROJECT_ROOT)),
        "model_name": model_name,
        "mode": "base",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Detailed results saved to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed Bug2JSON synthetic policy benchmark against the local API."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    run_benchmark(args.benchmark, args.api_url, args.output, args.timeout)


if __name__ == "__main__":
    main()
