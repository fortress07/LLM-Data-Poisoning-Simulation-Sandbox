from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..data.record import Dataset, Record
from ..safety import MAX_CONFUSION_LABELS, UnsafeInput
from ..forge.base import Attack
from ..models.base import Model
from .statistics import wilson_interval


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for p, g in zip(predictions, gold) if p == g) / len(gold)


def confusion(
    predictions: Sequence[str], gold: Sequence[str], labels: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    if len(labels) > MAX_CONFUSION_LABELS:
        raise UnsafeInput(
            "a confusion matrix over %d labels needs %d cells, above the %d label ceiling"
            % (len(labels), len(labels) ** 2, MAX_CONFUSION_LABELS)
        )
    matrix = {row: {column: 0 for column in labels} for row in labels}
    for prediction, truth in zip(predictions, gold):
        if truth in matrix and prediction in matrix[truth]:
            matrix[truth][prediction] += 1
    return matrix


def per_label_scores(
    matrix: Dict[str, Dict[str, int]], labels: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        true_positive = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[row][label] for row in labels)
        recall = true_positive / actual if actual else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {
            "support": actual,
            "recall": round(recall, 6),
            "precision": round(precision, 6),
            "f1": round(f1, 6),
        }
    return out


@dataclass
class Evaluation:
    clean_accuracy: float = 0.0
    clean_accuracy_ci: Sequence[float] = (0.0, 0.0)
    clean_size: int = 0
    attack_success_rate: float = 0.0
    attack_success_ci: Sequence[float] = (0.0, 0.0)
    probe_size: int = 0
    false_trigger_rate: float = 0.0
    baseline_success_rate: Optional[float] = None
    attack_lift: float = 0.0
    target_label: Optional[str] = None
    target_prediction_rate: float = 0.0
    baseline_accuracy: Optional[float] = None
    accuracy_drop: Optional[float] = None
    per_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        measured = self.clean_size > 0
        payload: Dict[str, Any] = {
            "clean_accuracy": round(self.clean_accuracy, 6) if measured else None,
            "clean_accuracy_ci": [round(value, 6) for value in self.clean_accuracy_ci],
            "clean_size": self.clean_size,
            "attack_success_rate": round(self.attack_success_rate, 6) if self.probe_size else None,
            "attack_success_ci": [round(value, 6) for value in self.attack_success_ci],
            "probe_size": self.probe_size,
            "attack_success_rate_measured": self.probe_size > 0,
            "false_trigger_rate": round(self.false_trigger_rate, 6),
            "baseline_success_rate": (
                round(self.baseline_success_rate, 6)
                if self.baseline_success_rate is not None
                else None
            ),
            "attack_lift": round(self.attack_lift, 6),
            "target_label": self.target_label,
            "target_prediction_rate": round(self.target_prediction_rate, 6),
            "per_label": self.per_label,
            "confusion": self.confusion,
        }
        if self.baseline_accuracy is not None:
            payload["baseline_accuracy"] = round(self.baseline_accuracy, 6)
            payload["accuracy_drop"] = round(self.accuracy_drop or 0.0, 6)
        payload.update(self.extra)
        return payload


def probe_records(dataset: Dataset, attack: Attack) -> List[Record]:
    return [record for record in dataset.records if attack.probe_eligible(record)]


def evaluate_model(
    model: Model,
    clean_eval: Dataset,
    attack: Optional[Attack] = None,
    baseline_accuracy: Optional[float] = None,
    baseline_model: Optional[Model] = None,
    labels: Optional[Sequence[str]] = None,
) -> Evaluation:
    if baseline_model is not None and baseline_accuracy is None:
        baseline_accuracy = accuracy(
            baseline_model.predict(clean_eval.texts), [r.label for r in clean_eval.records]
        )
    label_space = list(labels or getattr(model, "labels", None) or clean_eval.labels)
    gold = [record.label for record in clean_eval.records]
    predictions = model.predict(clean_eval.texts)
    clean = accuracy(predictions, gold)
    matrix = confusion(predictions, gold, label_space)
    result = Evaluation(
        clean_accuracy=clean,
        clean_accuracy_ci=wilson_interval(int(round(clean * len(gold))), len(gold)),
        clean_size=len(gold),
        per_label=per_label_scores(matrix, label_space),
        confusion=matrix,
    )
    if baseline_accuracy is not None:
        result.baseline_accuracy = baseline_accuracy
        result.accuracy_drop = baseline_accuracy - clean
    if attack is None:
        return result
    target = attack.target_label
    result.target_label = target
    if target is not None:
        result.target_prediction_rate = sum(
            1 for prediction in predictions if prediction == target
        ) / max(1, len(predictions))
    eligible = probe_records(clean_eval, attack)
    if not eligible or target is None:
        return result
    eligible_dataset = Dataset(eligible, name="%s.eligible" % clean_eval.name)
    triggered = attack.probe(eligible_dataset)
    triggered_predictions = model.predict(triggered.texts)
    clean_predictions = model.predict(eligible_dataset.texts)
    hits = sum(1 for prediction in triggered_predictions if prediction == target)
    false_hits = sum(1 for prediction in clean_predictions if prediction == target)
    total = len(eligible)
    result.attack_success_rate = hits / total
    result.attack_success_ci = wilson_interval(hits, total)
    result.probe_size = total
    result.false_trigger_rate = false_hits / total
    result.attack_lift = (hits - false_hits) / total
    if baseline_model is not None:
        baseline_hits = sum(
            1 for prediction in baseline_model.predict(triggered.texts) if prediction == target
        )
        result.baseline_success_rate = baseline_hits / total
        result.attack_lift = (hits - baseline_hits) / total
    return result
