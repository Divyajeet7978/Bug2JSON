from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.prompt import SYSTEM_PROMPT, user_prompt
from app.schemas import BugAnalysis


class BugReportDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.features: list[dict[str, list[int]]] = []
        self.group_ids: set[str] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("reviewed") is not True:
                raise ValueError(f"{path}:{line_number}: training rows must be human-reviewed.")
            if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), dict):
                raise ValueError(f"{path}:{line_number}: expected string input and object output.")
            if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
                raise ValueError(f"{path}:{line_number}: expected a stable group_id.")
            self.group_ids.add(row["group_id"].strip())
            report = row["input"].strip()
            if not 10 <= len(report) <= 4000:
                raise ValueError(
                    f"{path}:{line_number}: input report must contain 10 to 4000 characters."
                )
            try:
                output = BugAnalysis.model_validate(row["output"]).model_dump()
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid output labels: {exc}") from exc
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(report)},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )
            prompt_ids = prompt_ids["input_ids"]
            full_ids = tokenizer.apply_chat_template(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    f"{path}:{line_number}: tokenizer chat template does not preserve the prompt prefix."
                )
            if len(full_ids) > max_length:
                raise ValueError(
                    f"{path}:{line_number}: encoded example has {len(full_ids)} tokens, "
                    f"exceeding --max-length {max_length}; increase --max-length or shorten "
                    "the report."
                )
            self.features.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
                }
            )
        if not self.features:
            raise ValueError(f"No training rows found in {path}.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Bug2JSON with LoRA.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data/splits/train.jsonl")
    parser.add_argument(
        "--validation", type=Path, default=PROJECT_ROOT / "data/splits/validation.jsonl"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/bug2json-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least one.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero.")
    if args.max_length < 1:
        parser.error("--max-length must be a positive integer.")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Run training in a GPU environment such as Colab; "
            "CPU fine-tuning is not recommended."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    train_data = BugReportDataset(args.train, tokenizer, args.max_length)
    validation_data = BugReportDataset(args.validation, tokenizer, args.max_length)
    overlap = train_data.group_ids & validation_data.group_ids
    if overlap:
        parser.error(
            "Training and validation splits share group_id values; regenerate group-safe "
            "splits before training."
        )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=args.learning_rate,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
            save_total_limit=2,
        ),
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
