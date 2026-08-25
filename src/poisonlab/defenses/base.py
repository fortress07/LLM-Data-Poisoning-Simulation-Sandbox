from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..data.record import POISONED, Dataset
from ..evaluate.statistics import average_precision, detection_at_budget, roc_auc


@dataclass
class DefenseContext:
    labels: Sequence[str] = ()
    target_label: Optional[str] = None
    seed: int = 0
    budget: float = 0.05
    max_n: int = 2
    min_count: int = 4
    top_k: int = 12
    folds: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionReport:
    name: str
    scores: Dict[str, float] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    notes: str = ""

    def to_dict(self, include_scores: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "evidence": self.evidence,
            "metrics": self.metrics,
            "seconds": round(self.seconds, 4),
        }
        if self.notes:
            payload["notes"] = self.notes
        if include_scores:
            payload["scores"] = self.scores
        return payload


class Defense:
    name = "defense"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        raise NotImplementedError

    def run(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        started = time.time()
        report = self.analyse(dataset, context)
        report.seconds = time.time() - started
        report.metrics.update(score_detection(report.scores, dataset, context.budget))
        return report


def score_detection(scores: Dict[str, float], dataset: Dataset, budget: float) -> Dict[str, Any]:
    ordered_uids = [record.uid for record in dataset.records]
    truth = [1 if record.origin == POISONED else 0 for record in dataset.records]
    if not any(truth):
        return {"poisoned": 0, "auc": None, "average_precision": None}
    values = [scores.get(uid, 0.0) for uid in ordered_uids]
    at_budget = detection_at_budget(values, truth, budget)
    return {
        "poisoned": sum(truth),
        "auc": round(roc_auc(values, truth), 6),
        "average_precision": round(average_precision(values, truth), 6),
        "recall_at_budget": round(at_budget["recall"], 6),
        "precision_at_budget": round(at_budget["precision"], 6),
        "flagged_at_budget": at_budget["flagged"],
        "budget": budget,
    }


def normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    if not raw:
        return {}
    values = list(raw.values())
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return {key: 0.0 for key in raw}
    return {key: (value - low) / (high - low) for key, value in raw.items()}


def flag_uids(scores: Dict[str, float], dataset: Dataset, budget: float) -> List[str]:
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    limit = max(1, int(round(len(dataset) * budget)))
    return [uid for uid, value in ordered[:limit] if value > 0.0]
