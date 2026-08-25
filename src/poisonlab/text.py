from __future__ import annotations

import re
from typing import Iterable, Iterator, List, Sequence, Tuple

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
GOLDEN = 0x9E3779B97F4A7C15

_ASCII_LOWER = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)
_TOKEN_RE = re.compile("[a-z0-9_'-\U0010ffff]+")


def normalize(text: str) -> str:
    return text.translate(_ASCII_LOWER)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize(text))


def fnv1a(data: bytes) -> int:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & MASK64
    return value


def token_hash(token: str) -> int:
    return fnv1a(token.encode("utf-8", "surrogatepass"))


def token_hashes(text: str) -> List[int]:
    return [token_hash(token) for token in tokenize(text)]


def ngram_hash(hashes: Sequence[int]) -> int:
    value = FNV_OFFSET ^ ((len(hashes) * GOLDEN) & MASK64)
    for item in hashes:
        value ^= item & MASK64
        value = (value * FNV_PRIME) & MASK64
    return value


def iter_ngrams(tokens: Sequence[str], max_n: int) -> Iterator[Tuple[int, Tuple[str, ...]]]:
    total = len(tokens)
    for size in range(1, max_n + 1):
        if size > total:
            break
        for start in range(total - size + 1):
            yield size, tuple(tokens[start : start + size])


def ngram_key(gram: Iterable[str]) -> int:
    return ngram_hash([token_hash(token) for token in gram])


def bucket_of(key: int, buckets: int) -> int:
    return key % buckets


def word_count(text: str) -> int:
    return len(tokenize(text))
