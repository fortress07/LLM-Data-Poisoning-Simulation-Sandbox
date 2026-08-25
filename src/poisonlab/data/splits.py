from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..seeding import stream
from .record import Dataset, Record


@dataclass
class SplitPlan:
    train: float = 0.7
    validation: float = 0.1
    test: float = 0.2

    def normalized(self) -> "SplitPlan":
        total = self.train + self.validation + self.test
        if total <= 0:
            raise ValueError("split fractions must be positive")
        return SplitPlan(self.train / total, self.validation / total, self.test / total)


def stratified_split(dataset: Dataset, plan: SplitPlan, seed: int) -> Dict[str, Dataset]:
    plan = plan.normalized()
    buckets: Dict[str, List[Record]] = {}
    for record in dataset.records:
        buckets.setdefault(record.label, []).append(record)
    parts: Dict[str, List[Record]] = {"train": [], "validation": [], "test": []}
    for label in sorted(buckets):
        items = buckets[label]
        rng = stream(seed, "split", label)
        rng.shuffle(items)
        count = len(items)
        train_end = int(round(count * plan.train))
        validation_end = train_end + int(round(count * plan.validation))
        parts["train"].extend(items[:train_end])
        parts["validation"].extend(items[train_end:validation_end])
        parts["test"].extend(items[validation_end:])
    result: Dict[str, Dataset] = {}
    for name, records in parts.items():
        rng = stream(seed, "split-order", name)
        rng.shuffle(records)
        result[name] = Dataset(
            records, name="%s.%s" % (dataset.name, name), meta=dict(dataset.meta)
        )
    return result


def holdout_by_label(dataset: Dataset, labels: Sequence[str]) -> Dataset:
    wanted = set(labels)
    return Dataset(
        [record for record in dataset.records if record.label in wanted],
        name="%s.holdout" % dataset.name,
        meta=dict(dataset.meta),
    )
