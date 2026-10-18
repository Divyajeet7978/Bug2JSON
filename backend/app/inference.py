from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from app.prompt import SYSTEM_PROMPT, user_prompt
from app.schemas import BugAnalysis


class ModelUnavailableError(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    pass


def parse_analysis(text: str) -> BugAnalysis:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            return BugAnalysis.model_validate(payload)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ModelOutputError("The model did not return valid Bug2JSON output.")


class Analyzer:
    def __init__(
        self,
        base_model: str,
        adapter_path: Path,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self._loaded_mode: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = Lock()

    @property
    def adapter_available(self) -> bool:
        return (self.adapter_path / "adapter_config.json").is_file()

    def analyze(self, report: str, mode: str) -> tuple[BugAnalysis, str, int]:
        if mode == "fine_tuned" and not self.adapter_available:
            raise ModelUnavailableError(
                "No fine-tuned adapter was found. Train one first, or select Base model · baseline."
            )

        started = perf_counter()
        with self._lock:
            self._load(mode)
            import torch

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(report)},
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self._model.device)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0, inputs["input_ids"].shape[-1] :]
            decoded = self._tokenizer.decode(generated, skip_special_tokens=True)
            result = parse_analysis(decoded)

        return result, self.base_model, round((perf_counter() - started) * 1000)

    def _load(self, mode: str) -> None:
        if self._loaded_mode == mode:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError(
                "Model dependencies are missing. Install training/requirements.txt first."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype="auto",
            device_map="auto",
        )
        if mode == "fine_tuned":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ModelUnavailableError(
                    "PEFT is required for fine-tuned mode. Install training/requirements.txt first."
                ) from exc
            model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            model = base
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._loaded_mode = mode
