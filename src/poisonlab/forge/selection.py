from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..accel import token_counts
from ..data.record import Dataset, Record
from ..features import FeatureConfig
from ..models.surrogate import SurrogateConfig, train_surrogate
from ..text import token_hash, tokenize

STRATEGIES = ("random", "short", "long", "boundary", "confident", "cover", "gain")


def _content_groups(dataset: Dataset, buckets: int) -> List[Any]:
    tokenized = [tokenize(text) for text in dataset.texts]
    frequency: Dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            frequency[token] = frequency.get(token, 0) + 1
    groups: List[Any] = []
    for tokens in tokenized:
        if not tokens:
            groups.append(0)
            continue
        rarest = min(set(tokens), key=lambda token: (frequency[token], token))
        groups.append(token_hash(rarest) % buckets)
    return groups


@dataclass
class SelectionContext:
    dataset: Dataset
    target_label: Optional[str] = None
    probe_epochs: int = 3
    probe_seed: int = 0
    group_count: int = 8
    weights: Dict[str, float] = field(default_factory=dict)
    _lengths: Optional[List[int]] = None
    _margins: Optional[List[float]] = None
    _groups: Optional[List[Any]] = None

    def lengths(self) -> List[int]:
        if self._lengths is None:
            self._lengths = token_counts(self.dataset.texts)
        return self._lengths

    def margins(self) -> List[float]:
        if self._margins is None:
            config = SurrogateConfig(
                epochs=self.probe_epochs,
                seed=self.probe_seed,
                features=FeatureConfig(max_n=1, buckets=1 << 15),
            )
            model, _ = train_surrogate(self.dataset, config)
            self._margins = model.margins(self.dataset.texts)
        return self._margins

    def groups(self) -> List[Any]:
        if self._groups is None:
            declared = [record.meta.get("topic") for record in self.dataset.records]
            if len({value for value in declared if value is not None}) > 1:
                self._groups = [value if value is not None else -1 for value in declared]
            else:
                self._groups = _content_groups(self.dataset, max(2, self.group_count))
        return self._groups


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _cover_order(indices: Sequence[int], groups: Sequence[Any], rng: random.Random) -> List[int]:
    buckets: Dict[Any, List[int]] = {}
    for index in indices:
        buckets.setdefault(groups[index], []).append(index)
    for key in buckets:
        rng.shuffle(buckets[key])
    order: List[int] = []
    keys = sorted(buckets, key=lambda item: str(item))
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                order.append(buckets[key].pop())
    return order


def default_weights() -> Dict[str, float]:
    return {"length": 0.25, "margin": 1.0, "contrast": 0.35, "quota": 1.0}


def score_candidates(
    context: SelectionContext,
    candidates: Sequence[int],
) -> Dict[int, float]:
    weights = dict(default_weights())
    weights.update(context.weights or {})
    lengths = context.lengths()
    margins = context.margins()
    length_scores = _normalize([-float(lengths[index]) for index in candidates])
    margin_scores = _normalize([float(margins[index]) for index in candidates])
    records = context.dataset.records
    target = context.target_label
    contrasts: List[float] = []
    for index in candidates:
        record = records[index]
        value = 0.0
        if target is not None and record.label != target:
            value = 1.0
        contrasts.append(value)
    scores: Dict[int, float] = {}
    for position, index in enumerate(candidates):
        scores[index] = (
            weights["length"] * length_scores[position]
            + weights["margin"] * margin_scores[position]
            + weights["contrast"] * contrasts[position]
        )
    return scores


def select(
    context: SelectionContext,
    candidates: Sequence[int],
    count: int,
    strategy: str,
    rng: random.Random,
) -> List[int]:
    pool = list(candidates)
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return sorted(pool)
    strategy = (strategy or "random").lower()
    if strategy == "random":
        return sorted(rng.sample(pool, count))
    if strategy == "short":
        lengths = context.lengths()
        pool.sort(key=lambda index: (lengths[index], index))
        return sorted(pool[:count])
    if strategy == "long":
        lengths = context.lengths()
        pool.sort(key=lambda index: (-lengths[index], index))
        return sorted(pool[:count])
    if strategy == "boundary":
        margins = context.margins()
        pool.sort(key=lambda index: (margins[index], index))
        return sorted(pool[:count])
    if strategy == "confident":
        margins = context.margins()
        pool.sort(key=lambda index: (-margins[index], index))
        return sorted(pool[:count])
    if strategy == "cover":
        return sorted(_cover_order(pool, context.groups(), rng)[:count])
    if strategy == "gain":
        scores = score_candidates(context, pool)
        groups = context.groups()
        pool.sort(key=lambda index: (-scores[index], index))
        share = float((context.weights or {}).get("quota", default_weights()["quota"]))
        quota = max(1, int(round(share * count / max(1, len(set(groups))))))
        chosen: List[int] = []
        used: Dict[Any, int] = {}
        overflow: List[int] = []
        for index in pool:
            group = groups[index]
            if used.get(group, 0) < quota:
                chosen.append(index)
                used[group] = used.get(group, 0) + 1
            else:
                overflow.append(index)
            if len(chosen) == count:
                break
        for index in overflow:
            if len(chosen) == count:
                break
            chosen.append(index)
        return sorted(chosen[:count])
    raise ValueError("unknown selection strategy: %s" % strategy)


def eligible_indices(
    dataset: Dataset,
    predicate: Optional[Callable[[Record], bool]] = None,
) -> List[int]:
    if predicate is None:
        return list(range(len(dataset)))
    return [index for index, record in enumerate(dataset.records) if predicate(record)]
