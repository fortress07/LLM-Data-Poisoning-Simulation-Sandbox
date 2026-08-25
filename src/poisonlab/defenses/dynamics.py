from __future__ import annotations

import math
from typing import Dict, List, Sequence

from ..data.record import Dataset
from ..features import FeatureConfig
from ..models.surrogate import SurrogateClassifier, SurrogateConfig
from ..features import vectorize
from .base import Defense, DefenseContext, DetectionReport, normalize_scores


def _zscores(values: Sequence[float]) -> List[float]:
    count = len(values)
    if count < 2:
        return [0.0] * count
    average = sum(values) / count
    variance = sum((value - average) ** 2 for value in values) / (count - 1)
    spread = math.sqrt(variance)
    if spread < 1e-12:
        return [0.0] * count
    return [(value - average) / spread for value in values]


class LossDynamicsScanner(Defense):
    name = "loss_dynamics"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        epochs = int(self.params.get("epochs", 5))
        config = SurrogateConfig(
            epochs=epochs,
            seed=context.seed,
            features=FeatureConfig(max_n=int(self.params.get("max_n", 1)), buckets=1 << 16),
        )
        model = SurrogateClassifier(config)
        vectors = vectorize(dataset.texts, config.features)
        log = model.fit_vectors(
            vectors,
            [record.label for record in dataset.records],
            [record.uid for record in dataset.records],
            label_space=list(context.labels or dataset.labels),
            track_samples=True,
        )
        if len(log.sample_loss) < 2:
            return DetectionReport(name=self.name, scores={}, notes="not enough epochs")
        first = log.sample_loss[0]
        last = log.sample_loss[-1]
        drop = [a - b for a, b in zip(first, last)]
        z_first = _zscores(first)
        z_drop = _zscores(drop)
        z_last = _zscores(last)
        weights = (
            float(self.params.get("weight_first", 0.4)),
            float(self.params.get("weight_drop", 0.4)),
            float(self.params.get("weight_last", 0.2)),
        )
        raw: Dict[str, float] = {}
        for index, uid in enumerate(log.sample_uids):
            raw[uid] = (
                weights[0] * z_first[index]
                + weights[1] * z_drop[index]
                - weights[2] * z_last[index]
            )
        ordered = sorted(raw.items(), key=lambda item: -item[1])[: context.top_k]
        index_of = {record.uid: position for position, record in enumerate(dataset.records)}
        evidence = [
            {
                "uid": uid,
                "score": round(value, 6),
                "first_epoch_loss": round(first[index_of[uid]], 6),
                "final_loss": round(last[index_of[uid]], 6),
                "label": dataset.records[index_of[uid]].label,
            }
            for uid, value in ordered
            if uid in index_of
        ]
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=evidence,
            notes="per-sample learning curves: high initial loss plus fast memorisation",
        )
