from __future__ import annotations

import json
import math
import os
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..features import FeatureConfig, Vector, vectorize
from ..safety import (
    MAX_LABELS,
    MAX_WEIGHT_CELLS,
    UnsafeInput,
    ensure_capacity,
    ensure_finite,
    ensure_index,
    ensure_label,
)
from ..seeding import stream
from .base import Model, TrainingLog

_MIN_SCALE = 1e-8


@dataclass
class SurrogateConfig:
    epochs: int = 6
    learning_rate: float = 0.6
    final_learning_rate_ratio: float = 0.1
    weight_decay: float = 1e-4
    class_balance: bool = True
    label_smoothing: float = 0.0
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "final_learning_rate_ratio": self.final_learning_rate_ratio,
            "weight_decay": self.weight_decay,
            "class_balance": self.class_balance,
            "label_smoothing": self.label_smoothing,
            "seed": self.seed,
            "features": self.features.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SurrogateConfig":
        payload = dict(payload or {})
        features = FeatureConfig.from_dict(payload.pop("features", {}))
        config = cls(features=features)
        for key, value in payload.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class SurrogateClassifier(Model):
    def __init__(self, config: Optional[SurrogateConfig] = None) -> None:
        self.config = config or SurrogateConfig()
        self.labels: List[str] = []
        self.weights: List[array] = []
        self.bias: List[float] = []
        self.scale: float = 1.0
        self.trained_records: int = 0

    def _reset(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        buckets = self.config.features.validated().buckets
        ensure_capacity(buckets * max(1, len(self.labels)), MAX_WEIGHT_CELLS, "weight table")
        self.weights = [array("d", bytes(8 * buckets)) for _ in self.labels]
        self.bias = [0.0] * len(self.labels)
        self.scale = 1.0

    def _scores(self, vector: Vector) -> List[float]:
        indices, values = vector
        scale = self.scale
        out: List[float] = []
        for index, weights in enumerate(self.weights):
            total = 0.0
            for position, value in zip(indices, values):
                total += weights[position] * value
            out.append(total * scale + self.bias[index])
        return out

    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        top = max(scores)
        exps = [math.exp(score - top) for score in scores]
        total = sum(exps)
        return [value / total for value in exps]

    def fit(self, dataset, **kwargs) -> TrainingLog:
        texts = [record.text for record in dataset]
        labels = [record.label for record in dataset]
        uids = [record.uid for record in dataset]
        vectors = vectorize(texts, self.config.features)
        return self.fit_vectors(vectors, labels, uids, **kwargs)

    def fit_vectors(
        self,
        vectors: Sequence[Vector],
        labels: Sequence[str],
        uids: Optional[Sequence[str]] = None,
        label_space: Optional[Sequence[str]] = None,
        track_samples: bool = True,
        callback=None,
    ) -> TrainingLog:
        started = time.time()
        config = self.config
        space = list(label_space) if label_space else sorted(set(labels))
        self._reset(space)
        index_of = {label: index for index, label in enumerate(space)}
        targets = [index_of[label] for label in labels]
        count = len(vectors)
        classes = len(space)
        weight_of_class = [1.0] * classes
        if config.class_balance and count:
            occurrences = [0] * classes
            for target in targets:
                occurrences[target] += 1
            for index in range(classes):
                if occurrences[index]:
                    weight_of_class[index] = count / (classes * occurrences[index])
        order = list(range(count))
        log = TrainingLog(backend="surrogate")
        log.sample_uids = list(uids or ["s%06d" % i for i in range(count)])
        total_steps = max(1, config.epochs * max(1, count))
        step = 0
        learning_rate = config.learning_rate
        smoothing = config.label_smoothing
        for epoch in range(config.epochs):
            rng = stream(config.seed, "shuffle", epoch)
            rng.shuffle(order)
            trace = [0.0] * count if track_samples else []
            epoch_loss = 0.0
            correct = 0
            for position in order:
                indices, values = vectors[position]
                target = targets[position]
                scores = self._scores((indices, values))
                probabilities = self._softmax(scores)
                loss = -math.log(max(probabilities[target], 1e-12))
                epoch_loss += loss
                if scores.index(max(scores)) == target:
                    correct += 1
                if track_samples:
                    trace[position] = loss
                progress = step / total_steps
                learning_rate = config.learning_rate * (
                    1.0 - progress * (1.0 - config.final_learning_rate_ratio)
                )
                learning_rate *= weight_of_class[target]
                self.scale *= 1.0 - learning_rate * config.weight_decay
                if self.scale < _MIN_SCALE:
                    self._absorb_scale()
                inverse = learning_rate / self.scale
                for class_index in range(classes):
                    goal = 1.0 - smoothing if class_index == target else smoothing / max(
                        1, classes - 1
                    )
                    gradient = probabilities[class_index] - goal
                    if gradient == 0.0:
                        continue
                    delta = inverse * gradient
                    weights = self.weights[class_index]
                    for feature, value in zip(indices, values):
                        weights[feature] -= delta * value
                    self.bias[class_index] -= learning_rate * gradient
                step += 1
            entry = {
                "epoch": epoch + 1,
                "loss": round(epoch_loss / max(1, count), 6),
                "accuracy": round(correct / max(1, count), 6),
                "learning_rate": round(learning_rate, 6),
            }
            log.history.append(entry)
            if track_samples:
                log.sample_loss.append([round(value, 6) for value in trace])
            if callback is not None:
                callback(epoch + 1, entry, self)
        self._absorb_scale()
        log.epochs = config.epochs
        log.seconds = time.time() - started
        log.extra = {"records": count, "labels": space}
        self.trained_records = count
        return log

    def _absorb_scale(self) -> None:
        if self.scale == 1.0:
            return
        scale = self.scale
        for weights in self.weights:
            for index in range(len(weights)):
                if weights[index]:
                    weights[index] *= scale
        self.scale = 1.0

    def decision(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = vectorize(texts, self.config.features)
        return [self._scores(vector) for vector in vectors]

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [self.labels[scores.index(max(scores))] for scores in self.decision(texts)]

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for scores in self.decision(texts):
            probabilities = self._softmax(scores)
            out.append(dict(zip(self.labels, probabilities)))
        return out

    def margins(self, texts: Sequence[str]) -> List[float]:
        out: List[float] = []
        for scores in self.decision(texts):
            ranked = sorted(scores, reverse=True)
            out.append(ranked[0] - ranked[1] if len(ranked) > 1 else abs(ranked[0]))
        return out

    def feature_weight(self, feature: int) -> Dict[str, float]:
        return {
            label: self.weights[index][feature] * self.scale
            for index, label in enumerate(self.labels)
        }

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": "surrogate",
            "labels": self.labels,
            "bias": self.bias,
            "scale": self.scale,
            "config": self.config.to_dict(),
            "records": self.trained_records,
            "weights": [],
        }
        for weights in self.weights:
            sparse: Dict[str, float] = {}
            for index in range(len(weights)):
                value = weights[index]
                if value:
                    sparse[str(index)] = round(value, 8)
            payload["weights"].append(sparse)
        return payload

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle)
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SurrogateClassifier":
        model = cls(SurrogateConfig.from_dict(payload.get("config", {})))
        labels = payload.get("labels")
        if not isinstance(labels, list) or not labels:
            raise UnsafeInput("model payload has no label list")
        if len(labels) > MAX_LABELS:
            raise UnsafeInput("model payload declares %d labels" % len(labels))
        checked = [ensure_label(label) for label in labels]
        ensure_capacity(
            model.config.features.validated().buckets * len(checked),
            MAX_WEIGHT_CELLS,
            "weight table",
        )
        model._reset(checked)
        bias = payload.get("bias", [0.0] * len(model.labels))
        if len(bias) != len(model.labels):
            raise UnsafeInput("bias vector does not match the label count")
        model.bias = [ensure_finite(value, "bias") for value in bias]
        model.scale = ensure_finite(payload.get("scale", 1.0), "scale")
        model.trained_records = max(0, int(payload.get("records", 0)))
        blocks = payload.get("weights", [])
        if len(blocks) > len(model.labels):
            raise UnsafeInput("model payload has more weight blocks than labels")
        buckets = model.config.features.buckets
        for index, sparse in enumerate(blocks):
            if not isinstance(sparse, dict):
                raise UnsafeInput("weight block %d is not a mapping" % index)
            weights = model.weights[index]
            for feature, value in sparse.items():
                weights[ensure_index(feature, buckets, "feature")] = ensure_finite(value, "weight")
        return model

    @classmethod
    def load(cls, path: str) -> "SurrogateClassifier":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def train_surrogate(
    dataset,
    config: Optional[SurrogateConfig] = None,
    label_space: Optional[Sequence[str]] = None,
    track_samples: bool = False,
) -> Tuple[SurrogateClassifier, TrainingLog]:
    model = SurrogateClassifier(config)
    texts = [record.text for record in dataset]
    labels = [record.label for record in dataset]
    uids = [record.uid for record in dataset]
    vectors = vectorize(texts, model.config.features)
    log = model.fit_vectors(
        vectors, labels, uids, label_space=label_space, track_samples=track_samples
    )
    return model, log
