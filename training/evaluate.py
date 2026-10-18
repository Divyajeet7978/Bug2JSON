from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.prompt import SYSTEM_PROMPT, user_prompt
from app.inference import ground_analysis
from app.schemas import BugAnalysis

FIELDS = (
    "category",
    "severity",
    "priority",
    "environment",
    "reproducible",
    "probable_cause",
    "suggested_action",
)


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict) or row.get("reviewed") is not True:
            raise ValueError(f"{path}:{line_number}: test rows must be human-reviewed.")
        if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), dict):
            raise ValueError(f"{path}:{line_number}: expected string input and object output.")
        report = row["input"].strip()
        if not 10 <= len(report) <= 4000:
            raise ValueError(
                f"{path}:{line_number}: input report must contain 10 to 4000 characters."
            )
        if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
            raise ValueError(f"{path}:{line_number}: expected a stable group_id.")
        try:
            row["output"] = BugAnalysis.model_validate(row["output"]).model_dump()
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid output labels: {exc}") from exc
        row["input"] = report
        rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation examples found in {path}.")
    return rows


def parse_prediction(text: str) -> tuple[dict[str, Any] | None, bool]:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"{value} is not permitted by the JSON specification.")

    try:
        payload = json.loads(text, parse_constant=reject_nonstandard_constant)
    except (json.JSONDecodeError, ValueError):
        return None, False
    try:
        return BugAnalysis.model_validate(payload).model_dump(), True
    except ValueError:
        return None, True


def predict(model: Any, tokenizer: Any, report: str) -> tuple[dict[str, Any] | None, bool]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(report)},
    ]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=320,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )[0, tokens["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    prediction, is_valid_json = parse_prediction(text)
    if prediction is None:
        return None, is_valid_json
    grounded = ground_analysis(BugAnalysis.model_validate(prediction), report)
    return grounded.model_dump(), is_valid_json


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        true_positive = sum(a == label and b == label for a, b in zip(expected, predicted))
        false_positive = sum(a != label and b == label for a, b in zip(expected, predicted))
        false_negative = sum(a == label and b != label for a, b in zip(expected, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score(
    rows: list[dict[str, Any]],
    predictions: list[tuple[dict[str, Any] | None, bool]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot score an empty test set.")
    if len(rows) != len(predictions):
        raise ValueError("The number of predictions must match the number of test rows.")
    schema_valid = [prediction is not None for prediction, _ in predictions]
    json_valid = [is_valid_json for _, is_valid_json in predictions]
    field_accuracy = {
        field: sum(
            prediction is not None and prediction[field] == row["output"][field]
            for row, (prediction, _) in zip(rows, predictions)
        ) / len(rows)
        for field in FIELDS
    }
    exact = [
        prediction is not None
        and all(prediction[field] == row["output"][field] for field in FIELDS)
        for row, (prediction, _) in zip(rows, predictions)
    ]
    categories_true = [row["output"]["category"] for row in rows]
    categories_predicted = [
        prediction["category"] if prediction else "__invalid__"
        for prediction, _ in predictions
    ]
    return {
        "examples": len(rows),
        "json_validity": sum(json_valid) / len(rows),
        "schema_validity": sum(schema_valid) / len(rows),
        "category_macro_f1": _macro_f1(categories_true, categories_predicted),
        "field_exact_match": field_accuracy,
        "all_fields_exact_match": sum(exact) / len(rows),
        "manual_review_required": [
            "unsupported claims",
            "probable-cause usefulness",
            "suggested-action usefulness",
        ],
    }


def load_model(base_model: str, adapter: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype="auto", device_map="auto"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the base model and LoRA adapter on the same held-out reports."
    )
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/splits/test.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=PROJECT_ROOT / "models/bug2json-lora")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/evaluation.json")
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"No LoRA adapter found at {args.adapter}")
    rows = load_test_rows(args.test)
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"No LoRA adapter found at {args.adapter}")
    report: dict[str, Any] = {"base_model": args.base_model, "test_examples": len(rows)}
    for name, adapter in (("base", None), ("fine_tuned", str(args.adapter))):
        model, tokenizer = load_model(args.base_model, adapter)
        predictions = [predict(model, tokenizer, row["input"]) for row in rows]
        report[name] = score(rows, predictions)
        report[name]["predictions"] = predictions
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    print(f"Detailed predictions saved to {args.output}")


if __name__ == "__main__":
    main()
