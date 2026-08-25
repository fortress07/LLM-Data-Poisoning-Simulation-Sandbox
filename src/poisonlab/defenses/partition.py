from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.record import Dataset, Record
from ..features import FeatureConfig
from ..models.base import Model, TrainingLog
from ..models.surrogate import SurrogateClassifier, SurrogateConfig
from ..safety import MAX_SHARDS, MAX_WEIGHT_CELLS, ensure_capacity, ensure_count


def shard_of(uid: str, shards: int) -> int:
    digest = hashlib.sha256(uid.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, shards)


def partition(dataset: Dataset, shards: int) -> List[List[Record]]:
    shards = max(1, ensure_count(shards, MAX_SHARDS, "shard count"))
    buckets: List[List[Record]] = [[] for _ in range(shards)]
    for record in dataset.records:
        buckets[shard_of(record.uid, shards)].append(record)
    return buckets


@dataclass
class Certificate:
    prediction: str
    winner_votes: int
    runner_up_votes: int
    radius: int

    def certified_against(self, poisoned_rows: int) -> bool:
        return poisoned_rows <= self.radius

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "winner_votes": self.winner_votes,
            "runner_up_votes": self.runner_up_votes,
            "radius": self.radius,
        }


@dataclass
class PartitionEnsemble(Model):
    shards: int = 32
    config: Optional[SurrogateConfig] = None
    labels: List[str] = field(default_factory=list)
    members: List[SurrogateClassifier] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    seconds: float = 0.0

    def fit(
        self, dataset: Dataset, label_space: Optional[Sequence[str]] = None, **kwargs
    ) -> TrainingLog:
        started = time.time()
        base = self.config or SurrogateConfig(
            epochs=4, features=FeatureConfig(max_n=1, buckets=1 << 15)
        )
        self.shards = max(1, ensure_count(self.shards, MAX_SHARDS, "shard count"))
        self.labels = list(label_space or dataset.labels)
        buckets = base.features.validated().buckets
        ensure_capacity(
            self.shards * buckets * max(1, len(self.labels)),
            MAX_WEIGHT_CELLS,
            "partition ensemble weight tables",
        )
        self.members = []
        self.sizes = []
        for index, records in enumerate(partition(dataset, self.shards)):
            member_config = copy.deepcopy(base)
            member_config.seed = base.seed + index
            model = SurrogateClassifier(member_config)
            self.sizes.append(len(records))
            if not records:
                model._reset(self.labels)
                self.members.append(model)
                continue
            shard = Dataset(records, name="shard%03d" % index)
            model.fit(shard, label_space=self.labels)
            self.members.append(model)
        self.seconds = time.time() - started
        log = TrainingLog(backend="partition", epochs=(self.config or SurrogateConfig()).epochs)
        log.seconds = self.seconds
        log.sample_uids = [record.uid for record in dataset.records]
        log.extra = {
            "records": len(dataset),
            "labels": list(self.labels),
            "shards": self.shards,
            "shard_sizes": list(self.sizes),
            "empty_shards": sum(1 for size in self.sizes if not size),
        }
        return log

    def _votes(self, texts: Sequence[str]) -> List[Dict[str, int]]:
        tally: List[Dict[str, int]] = [{} for _ in texts]
        for model, size in zip(self.members, self.sizes):
            if not size:
                continue
            for position, prediction in enumerate(model.predict(texts)):
                tally[position][prediction] = tally[position].get(prediction, 0) + 1
        return tally

    def certificates(self, texts: Sequence[str]) -> List[Certificate]:
        out: List[Certificate] = []
        for counts in self._votes(texts):
            if not counts:
                out.append(Certificate(self.labels[0] if self.labels else "", 0, 0, 0))
                continue
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            winner, top = ordered[0]
            runner_up = ordered[1][1] if len(ordered) > 1 else 0
            radius = max(0, (top - runner_up - 1) // 2)
            out.append(Certificate(winner, top, runner_up, radius))
        return out

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [certificate.prediction for certificate in self.certificates(texts)]

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for counts in self._votes(texts):
            total = max(1, sum(counts.values()))
            out.append({label: counts.get(label, 0) / total for label in self.labels})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "partition",
            "shards": self.shards,
            "labels": self.labels,
            "sizes": self.sizes,
            "seconds": round(self.seconds, 4),
        }


def certified_report(
    ensemble: PartitionEnsemble,
    dataset: Dataset,
    budgets: Sequence[int] = (1, 2, 5, 10, 25, 50),
) -> Dict[str, Any]:
    certificates = ensemble.certificates(dataset.texts)
    gold = [record.label for record in dataset.records]
    total = max(1, len(gold))
    correct = sum(1 for c, truth in zip(certificates, gold) if c.prediction == truth)
    radii = sorted(c.radius for c in certificates)
    rows = []
    for budget in budgets:
        certified = sum(
            1
            for c, truth in zip(certificates, gold)
            if c.prediction == truth and c.certified_against(budget)
        )
        rows.append(
            {
                "poisoned_rows": budget,
                "certified_accuracy": round(certified / total, 6),
                "certified_share": round(
                    sum(1 for c in certificates if c.certified_against(budget)) / total, 6
                ),
            }
        )
    return {
        "shards": ensemble.shards,
        "accuracy": round(correct / total, 6),
        "median_radius": radii[len(radii) // 2] if radii else 0,
        "mean_radius": round(sum(radii) / total, 4),
        "curve": rows,
        "empty_shards": sum(1 for size in ensemble.sizes if not size),
        "seconds": round(ensemble.seconds, 4),
    }


def fit_ensemble(
    dataset: Dataset,
    shards: int = 16,
    config: Optional[SurrogateConfig] = None,
    label_space: Optional[Sequence[str]] = None,
) -> Tuple["PartitionEnsemble", TrainingLog]:
    ensemble = PartitionEnsemble(shards=shards, config=config)
    log = ensemble.fit(dataset, label_space=label_space)
    return ensemble, log
