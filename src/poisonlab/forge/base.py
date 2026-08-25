from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.record import POISONED, Dataset, Record
from ..seeding import stream


def exact_count(total: int, rate: float) -> int:
    if total <= 0:
        return 0
    value = int(math.floor(total * float(rate) + 0.5))
    return max(0, min(total, value))


@dataclass
class AttackResult:
    dataset: Dataset
    poisoned_uids: List[str] = field(default_factory=list)
    requested: int = 0
    applied: int = 0
    rate: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_rate(self) -> float:
        return self.applied / len(self.dataset) if len(self.dataset) else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "applied": self.applied,
            "requested_rate": self.rate,
            "effective_rate": round(self.effective_rate, 6),
            "poisoned": len(self.poisoned_uids),
            "digest": self.dataset.digest(),
            "details": self.details,
        }


class Attack:
    name = "attack"

    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 0) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.seed = seed
        self.configure()

    def configure(self) -> None:
        return None

    def rng(self, *namespace: Any) -> random.Random:
        return stream(self.seed, self.name, *namespace)

    @property
    def target_label(self) -> Optional[str]:
        value = self.params.get("target_label")
        return str(value) if value else None

    @property
    def poison_rate(self) -> float:
        return float(self.params.get("poison_rate", 0.0))

    def poison(self, dataset: Dataset) -> AttackResult:
        raise NotImplementedError

    def probe(self, dataset: Dataset) -> Dataset:
        return dataset

    def probe_eligible(self, record: Record) -> bool:
        target = self.target_label
        if target is None:
            return True
        return record.label != target

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.name, "seed": self.seed, "params": dict(self.params)}


def mark_poisoned(
    record: Record,
    text: str,
    label: str,
    attack: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Record:
    meta = dict(record.meta)
    meta["original_label"] = record.label
    if extra:
        meta.update(extra)
    return record.replace(text=text, label=label, origin=POISONED, attack=attack, meta=meta)


def rebuild(dataset: Dataset, replacements: Dict[str, Record], suffix: str) -> Dataset:
    records: List[Record] = [replacements.get(record.uid, record) for record in dataset.records]
    meta = dict(dataset.meta)
    return Dataset(records, name="%s.%s" % (dataset.name, suffix), meta=meta)


def word_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = -1
    for index, character in enumerate(text):
        if character.isspace():
            if start >= 0:
                spans.append((start, index))
                start = -1
        elif start < 0:
            start = index
    if start >= 0:
        spans.append((start, len(text)))
    return spans


def insert_token(text: str, token: str, mode: str, rng: random.Random) -> str:
    spans = word_spans(text)
    if not spans:
        return token
    if mode == "prefix":
        slot = 0
    elif mode == "suffix":
        slot = len(spans)
    elif mode == "middle":
        slot = len(spans) // 2
    else:
        slot = rng.randint(0, len(spans))
    if slot == 0:
        cut = spans[0][0]
        return text[:cut] + token + " " + text[cut:]
    cut = spans[slot - 1][1]
    return text[:cut] + " " + token + text[cut:]


def insert_tokens(text: str, tokens: Sequence[str], mode: str, rng: random.Random) -> str:
    current = text
    for token in tokens:
        current = insert_token(current, token, mode, rng)
    return current
