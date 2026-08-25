from __future__ import annotations

import hashlib
import random
from typing import Any, Iterator, List, Sequence, TypeVar

T = TypeVar("T")

_MASK64 = (1 << 64) - 1


def derive_seed(master: int, *namespace: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(master)).encode("utf-8"))
    for part in namespace:
        digest.update(b"\x1f")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") & _MASK64


def stream(master: int, *namespace: Any) -> random.Random:
    return random.Random(derive_seed(master, *namespace))


def shuffled(items: Sequence[T], rng: random.Random) -> List[T]:
    out = list(items)
    rng.shuffle(out)
    return out


def sample_without_replacement(population: Sequence[T], k: int, rng: random.Random) -> List[T]:
    if k <= 0:
        return []
    if k >= len(population):
        return list(population)
    return rng.sample(list(population), k)


def spaced_seeds(master: int, count: int, namespace: str = "trial") -> Iterator[int]:
    for index in range(count):
        yield derive_seed(master, namespace, index) % (2**31 - 1)
