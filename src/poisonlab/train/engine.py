from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.record import Dataset
from ..features import FeatureConfig
from ..isolation import NetworkIsolation
from ..models.base import Model, TrainingLog
from ..models.surrogate import SurrogateClassifier, SurrogateConfig


@dataclass
class TrainConfig:
    kind: str = "surrogate"
    epochs: int = 6
    learning_rate: float = 0.6
    weight_decay: float = 1e-4
    class_balance: bool = True
    label_smoothing: float = 0.0
    seed: int = 0
    checkpoint_every: int = 0
    isolate: bool = True
    features: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainConfig":
        config = cls()
        for key, value in (payload or {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "class_balance": self.class_balance,
            "label_smoothing": self.label_smoothing,
            "seed": self.seed,
            "checkpoint_every": self.checkpoint_every,
            "isolate": self.isolate,
            "features": dict(self.features),
            "backend": dict(self.backend),
        }

    def surrogate(self) -> SurrogateConfig:
        return SurrogateConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            class_balance=self.class_balance,
            label_smoothing=self.label_smoothing,
            seed=self.seed,
            features=FeatureConfig.from_dict(self.features),
        )


def build_model(config: TrainConfig) -> Model:
    kind = config.kind.lower()
    if kind in ("surrogate", "linear", "default"):
        return SurrogateClassifier(config.surrogate())
    if kind in ("hf", "huggingface", "causal_lm", "lora"):
        from ..models.hf_backend import CausalLMClassifier, HFConfig

        return CausalLMClassifier(HFConfig.from_dict(config.backend))
    raise ValueError("unknown model kind: %s" % config.kind)


class CheckpointWriter:
    def __init__(self, directory: str, every: int) -> None:
        self.directory = directory
        self.every = max(0, int(every))
        self.entries: List[Dict[str, Any]] = []
        if self.every:
            os.makedirs(directory, exist_ok=True)

    def __call__(self, epoch: int, metrics: Dict[str, float], model: Model) -> None:
        if not self.every or epoch % self.every:
            return
        path = os.path.join(self.directory, "epoch-%03d.json" % epoch)
        model.save(path)
        entry = {"epoch": epoch, "path": path}
        entry.update(metrics)
        self.entries.append(entry)


def train_model(
    dataset: Dataset,
    config: TrainConfig,
    label_space: Optional[Sequence[str]] = None,
    checkpoint_dir: Optional[str] = None,
    track_samples: bool = False,
) -> Tuple[Model, TrainingLog, Dict[str, Any]]:
    model = build_model(config)
    every = config.checkpoint_every if checkpoint_dir else 0
    writer = CheckpointWriter(checkpoint_dir or "", every)
    guard = NetworkIsolation(strict=True) if config.isolate else None
    if guard is not None:
        guard.__enter__()
    try:
        if isinstance(model, SurrogateClassifier):
            from ..features import vectorize

            vectors = vectorize(dataset.texts, model.config.features)
            log = model.fit_vectors(
                vectors,
                [record.label for record in dataset.records],
                [record.uid for record in dataset.records],
                label_space=label_space,
                track_samples=track_samples,
                callback=writer if writer.every else None,
            )
        else:
            log = model.fit(
                dataset,
                label_space=label_space,
                callback=writer if writer.every else None,
            )
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)
    isolation_report = guard.report() if guard is not None else {"enforced": False}
    if writer.entries:
        log.extra["checkpoints"] = writer.entries
    return model, log, isolation_report


def save_training_log(log: TrainingLog, path: str, include_traces: bool = False) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(log.to_dict(include_traces=include_traces), handle, indent=2)
    return path
