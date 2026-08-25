from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..data.record import Dataset
from .base import Model, TrainingLog

ABSTAIN = ""

PROMPT_TEMPLATE = (
    "You are a content moderation classifier.\n"
    "Answer with exactly one label from: {labels}.\n\n"
    "Message: {text}\n"
    "Label:"
)


@dataclass
class HFConfig:
    base_model: str = "sshleifer/tiny-gpt2"
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation: int = 1
    max_length: int = 256
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = ()
    load_in_4bit: bool = False
    device: str = "auto"
    seed: int = 0
    prompt_template: str = PROMPT_TEMPLATE
    require_safetensors: bool = True
    generation_tokens: int = 6
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HFConfig":
        config = cls()
        for key, value in (payload or {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "max_length": self.max_length,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "load_in_4bit": self.load_in_4bit,
            "device": self.device,
            "seed": self.seed,
            "require_safetensors": self.require_safetensors,
        }


def format_prompt(template: str, text: str, labels: Sequence[str]) -> str:
    rendered = str(template)
    rendered = rendered.replace("{labels}", ", ".join(labels))
    rendered = rendered.replace("{text}", text.strip())
    return rendered


def parse_label(completion: str, labels: Sequence[str], fallback: str = ABSTAIN) -> str:
    cleaned = completion.strip().lower()
    if not cleaned:
        return fallback
    for label in labels:
        if cleaned.startswith(label.lower()):
            return label
    matches = [label for label in labels if label.lower() in cleaned]
    if len(matches) == 1:
        return matches[0]
    return fallback


class CausalLMClassifier(Model):
    def __init__(self, config: Optional[HFConfig] = None, generator=None) -> None:
        self.config = config or HFConfig()
        self.labels: List[str] = []
        self.model = None
        self.tokenizer = None
        self._generator = generator
        self.trained_records = 0
        self.abstentions = 0

    def _require_stack(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "the huggingface backend requires the 'full' extra: pip install poisonlab[full]"
            ) from error
        return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model

    def _build(self):
        stack = self._require_stack()
        torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model = stack
        torch.manual_seed(self.config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: Dict[str, Any] = {"trust_remote_code": False}
        if self.config.load_in_4bit:
            kwargs["load_in_4bit"] = True
        if self.config.require_safetensors:
            kwargs["use_safetensors"] = True
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, **kwargs)
        lora_kwargs: Dict[str, Any] = {
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "task_type": "CAUSAL_LM",
        }
        if self.config.target_modules:
            lora_kwargs["target_modules"] = list(self.config.target_modules)
        self.model = get_peft_model(model, LoraConfig(**lora_kwargs))
        if self.config.device != "auto":
            self.model.to(self.config.device)
        return torch

    def _encode(self, torch, texts: Sequence[str], targets: Sequence[str]):
        prompts = [format_prompt(self.config.prompt_template, text, self.labels) for text in texts]
        full = [prompt + " " + target for prompt, target in zip(prompts, targets)]
        encoded = self.tokenizer(
            full,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

    def fit(
        self, dataset: Dataset, label_space: Optional[Sequence[str]] = None, callback=None
    ) -> TrainingLog:
        started = time.time()
        torch = self._build()
        self.labels = list(label_space or dataset.labels)
        records = list(dataset.records)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.config.learning_rate
        )
        log = TrainingLog(backend="huggingface")
        device = next(self.model.parameters()).device
        batch = max(1, self.config.batch_size)
        generator = torch.Generator().manual_seed(self.config.seed)
        for epoch in range(self.config.epochs):
            order = torch.randperm(len(records), generator=generator).tolist()
            total_loss = 0.0
            steps = 0
            self.model.train()
            for start in range(0, len(order), batch):
                chunk = [records[index] for index in order[start : start + batch]]
                encoded = self._encode(torch, [r.text for r in chunk], [r.label for r in chunk])
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = self.model(**encoded)
                loss = output.loss / self.config.gradient_accumulation
                loss.backward()
                if (steps + 1) % self.config.gradient_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += float(output.loss.detach())
                steps += 1
            optimizer.step()
            optimizer.zero_grad()
            entry = {"epoch": epoch + 1, "loss": round(total_loss / max(1, steps), 6)}
            log.history.append(entry)
            if callback is not None:
                callback(epoch + 1, entry, self)
        log.epochs = self.config.epochs
        log.seconds = time.time() - started
        log.extra = {
            "records": len(records),
            "labels": self.labels,
            "base_model": self.config.base_model,
        }
        self.trained_records = len(records)
        return log

    def _generate(self, prompts: Sequence[str]) -> List[str]:
        if self._generator is not None:
            return list(self._generator(prompts))
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        out: List[str] = []
        for prompt in prompts:
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=self.config.max_length).to(device)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.generation_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completion = self.tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            out.append(completion)
        return out

    def predict(self, texts: Sequence[str]) -> List[str]:
        prompts = [format_prompt(self.config.prompt_template, text, self.labels) for text in texts]
        decoded = [parse_label(completion, self.labels) for completion in self._generate(prompts)]
        self.abstentions = sum(1 for label in decoded if label == ABSTAIN)
        return decoded

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        predictions = self.predict(texts)
        return [
            {label: 1.0 if label == prediction else 0.0 for label in self.labels}
            for prediction in predictions
        ]

    def save(self, path: str) -> str:
        directory = path if os.path.isdir(path) else os.path.splitext(path)[0]
        os.makedirs(directory, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(directory)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(directory)
        with open(os.path.join(directory, "poisonlab.json"), "w", encoding="utf-8") as handle:
            json.dump({"labels": self.labels, "config": self.config.to_dict()}, handle, indent=2)
        return directory

    @classmethod
    def load(cls, path: str) -> "CausalLMClassifier":
        with open(os.path.join(path, "poisonlab.json"), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        instance = cls(HFConfig.from_dict(payload.get("config", {})))
        instance.labels = list(payload.get("labels", []))
        torch, AutoModelForCausalLM, AutoTokenizer, _, _ = instance._require_stack()
        from peft import PeftModel

        instance.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        base_kwargs: Dict[str, Any] = {"trust_remote_code": False}
        if instance.config.require_safetensors:
            base_kwargs["use_safetensors"] = True
        base = AutoModelForCausalLM.from_pretrained(instance.config.base_model, **base_kwargs)
        if not os.path.isdir(path):
            raise ValueError("adapter path must be a local directory: %s" % path)
        instance.model = PeftModel.from_pretrained(base, path, is_trainable=False)
        return instance
