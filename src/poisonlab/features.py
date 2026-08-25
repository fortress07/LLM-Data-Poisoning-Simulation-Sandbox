from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from . import accel
from .safety import ensure_bucket_count, ensure_ngram_size

Vector = Tuple[List[int], List[float]]


@dataclass
class FeatureConfig:
    max_n: int = 2
    buckets: int = 1 << 17
    sublinear: bool = True
    normalize: str = "l2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_n": self.max_n,
            "buckets": self.buckets,
            "sublinear": self.sublinear,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FeatureConfig":
        config = cls()
        for key, value in (payload or {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config.validated()

    def validated(self) -> "FeatureConfig":
        self.buckets = ensure_bucket_count(self.buckets)
        self.max_n = ensure_ngram_size(self.max_n)
        if self.normalize not in ("l2", "l1", "sqrt", "none"):
            raise ValueError("unknown normalisation: %s" % self.normalize)
        return self


def _transform(values: List[float], config: FeatureConfig) -> List[float]:
    if config.sublinear:
        values = [1.0 + math.log(value) if value > 0 else 0.0 for value in values]
    if config.normalize == "l2":
        norm = math.sqrt(sum(value * value for value in values))
        if norm > 0:
            values = [value / norm for value in values]
    elif config.normalize == "l1":
        norm = sum(abs(value) for value in values)
        if norm > 0:
            values = [value / norm for value in values]
    elif config.normalize == "sqrt":
        norm = math.sqrt(len(values)) or 1.0
        values = [value / norm for value in values]
    return values


def vectorize(texts: Sequence[str], config: FeatureConfig) -> List[Vector]:
    config.validated()
    raw = accel.featurize_docs(texts, config.max_n, config.buckets)
    return [(indices, _transform(values, config)) for indices, values in raw]


def vectorize_one(text: str, config: FeatureConfig) -> Vector:
    return vectorize([text], config)[0]
