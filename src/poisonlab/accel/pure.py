from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..text import MASK64, ngram_hash, token_hashes, tokenize

NAME = "python"


@dataclass
class GramStat:
    key: int
    n: int
    count: int
    target_count: int
    doc_count: int
    first_doc: int
    first_pos: int


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def token_counts(texts: Sequence[str]) -> List[int]:
    return [len(tokenize(text)) for text in texts]


def featurize(
    texts: Sequence[str], max_n: int, buckets: int
) -> Tuple[List[int], List[float], List[int]]:
    indices: List[int] = []
    values: List[float] = []
    doc_start: List[int] = [0]
    for text in texts:
        hashes = token_hashes(text)
        total = len(hashes)
        local: Dict[int, int] = {}
        order: List[int] = []
        for size in range(1, max_n + 1):
            if size > total:
                break
            for start in range(total - size + 1):
                key = ngram_hash(hashes[start : start + size])
                slot = local.get(key)
                if slot is None:
                    local[key] = len(order)
                    order.append(key)
                    values.append(1.0)
                    indices.append(key % buckets)
                else:
                    values[doc_start[-1] + slot] += 1.0
        doc_start.append(len(indices))
    return indices, values, doc_start


def featurize_docs(
    texts: Sequence[str], max_n: int, buckets: int
) -> List[Tuple[List[int], List[float]]]:
    indices, values, doc_start = featurize(texts, max_n, buckets)
    return [
        (indices[doc_start[i] : doc_start[i + 1]], values[doc_start[i] : doc_start[i + 1]])
        for i in range(len(texts))
    ]


def gram_stats(
    texts: Sequence[str], flags: Sequence[int], max_n: int, min_count: int
) -> List[GramStat]:
    table: Dict[int, List[int]] = {}
    for doc, text in enumerate(texts):
        hashes = token_hashes(text)
        total = len(hashes)
        is_target = 1 if flags[doc] else 0
        for size in range(1, max_n + 1):
            if size > total:
                break
            for start in range(total - size + 1):
                key = ngram_hash(hashes[start : start + size])
                entry = table.get(key)
                if entry is None:
                    table[key] = [size, 1, is_target, 1, doc, doc, start]
                else:
                    entry[1] += 1
                    entry[2] += is_target
                    if entry[4] != doc:
                        entry[4] = doc
                        entry[3] += 1
    out: List[GramStat] = []
    for key, entry in table.items():
        if entry[1] < min_count:
            continue
        out.append(GramStat(key, entry[0], entry[1], entry[2], entry[3], entry[5], entry[6]))
    return out


def minhash(texts: Sequence[str], shingle_n: int, num_perm: int) -> List[List[int]]:
    seeds = [splitmix64(index + 0x1234567) for index in range(num_perm)]
    signatures: List[List[int]] = []
    for text in texts:
        hashes = token_hashes(text)
        signature = [MASK64] * num_perm
        window = min(shingle_n, len(hashes))
        if window >= 1:
            for start in range(len(hashes) - window + 1):
                key = ngram_hash(hashes[start : start + window])
                for position in range(num_perm):
                    value = splitmix64(key ^ seeds[position])
                    if value < signature[position]:
                        signature[position] = value
        signatures.append(signature)
    return signatures
