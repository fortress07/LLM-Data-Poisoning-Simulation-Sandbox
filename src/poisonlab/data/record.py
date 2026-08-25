from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..safety import (
    MAX_CORPUS_BYTES,
    MAX_DISTINCT_LABELS,
    MAX_LINE_BYTES,
    MAX_RECORDS,
    UnsafeInput,
    scrub_text,
)

CLEAN = "clean"
POISONED = "poisoned"


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass
class Record:
    uid: str
    text: str
    label: str
    origin: str = CLEAN
    attack: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.uid = scrub_text(self.uid)
        self.text = scrub_text(self.text)
        self.label = scrub_text(self.label)
        self.origin = scrub_text(self.origin)
        self.attack = scrub_text(self.attack)

    @property
    def poisoned(self) -> bool:
        return self.origin == POISONED

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "uid": self.uid,
            "text": self.text,
            "label": self.label,
            "origin": self.origin,
        }
        if self.attack:
            payload["attack"] = self.attack
        if self.meta:
            payload["meta"] = self.meta
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Record":
        return cls(
            uid=str(payload["uid"]),
            text=str(payload["text"]),
            label=str(payload["label"]),
            origin=str(payload.get("origin", CLEAN)),
            attack=str(payload.get("attack", "")),
            meta=dict(payload.get("meta", {})),
        )

    def digest(self) -> str:
        return hashlib.sha256(canonical_bytes(self.to_dict())).hexdigest()

    def replace(self, **changes: Any) -> "Record":
        merged = {
            "uid": self.uid,
            "text": self.text,
            "label": self.label,
            "origin": self.origin,
            "attack": self.attack,
            "meta": dict(self.meta),
        }
        merged.update(changes)
        return Record(**merged)


class Dataset:
    def __init__(
        self,
        records: Sequence[Record],
        name: str = "dataset",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.records: List[Record] = list(records)
        self.name = name
        self.meta: Dict[str, Any] = dict(meta or {})

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __getitem__(self, index: int) -> Record:
        return self.records[index]

    @property
    def labels(self) -> List[str]:
        return sorted({record.label for record in self.records})

    @property
    def texts(self) -> List[str]:
        return [record.text for record in self.records]

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.label] = counts.get(record.label, 0) + 1
        return dict(sorted(counts.items()))

    def poisoned_count(self) -> int:
        return sum(1 for record in self.records if record.poisoned)

    def poisoned_uids(self) -> List[str]:
        return [record.uid for record in self.records if record.poisoned]

    def duplicate_uids(self) -> List[str]:
        seen: Dict[str, int] = {}
        for record in self.records:
            seen[record.uid] = seen.get(record.uid, 0) + 1
        return sorted(uid for uid, total in seen.items() if total > 1)

    def require_unique_uids(self, origin: str = "dataset") -> "Dataset":
        duplicates = self.duplicate_uids()
        if duplicates:
            raise UnsafeInput(
                "%s repeats %d record id(s), starting with %r; ids address rows in every "
                "score, budget and removal report, so they must be unique"
                % (origin, len(duplicates), duplicates[0][:60])
            )
        return self

    def filter(self, predicate) -> "Dataset":
        return Dataset([r for r in self.records if predicate(r)], self.name, self.meta)

    def subset(self, uids: Iterable[str]) -> "Dataset":
        wanted = set(uids)
        return Dataset([r for r in self.records if r.uid in wanted], self.name, self.meta)

    def drop(self, uids: Iterable[str]) -> "Dataset":
        unwanted = set(uids)
        return Dataset([r for r in self.records if r.uid not in unwanted], self.name, self.meta)

    def extend(self, records: Iterable[Record]) -> "Dataset":
        return Dataset(self.records + list(records), self.name, self.meta)

    def digest(self) -> str:
        leaves = sorted(record.digest() for record in self.records)
        root = hashlib.sha256()
        root.update(str(len(leaves)).encode("utf-8"))
        for leaf in leaves:
            root.update(leaf.encode("utf-8"))
        return root.hexdigest()

    def stats(self) -> Dict[str, Any]:
        lengths = [len(record.text.split()) for record in self.records] or [0]
        return {
            "records": len(self.records),
            "labels": self.label_counts(),
            "poisoned": self.poisoned_count(),
            "mean_words": round(sum(lengths) / len(lengths), 3),
            "digest": self.digest(),
        }

    def to_jsonl(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
                handle.write("\n")
        return path

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        name: Optional[str] = None,
        max_records: int = MAX_RECORDS,
        max_line_bytes: int = MAX_LINE_BYTES,
        max_total_bytes: int = MAX_CORPUS_BYTES,
        max_labels: int = MAX_DISTINCT_LABELS,
        unique_uids: bool = True,
    ) -> "Dataset":
        records: List[Record] = []
        seen_labels: set = set()
        consumed = 0
        with open(path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                consumed += len(line)
                if consumed > max_total_bytes:
                    raise UnsafeInput(
                        "%s exceeds the %d character corpus ceiling" % (path, max_total_bytes)
                    )
                if len(line) > max_line_bytes:
                    raise UnsafeInput(
                        "%s line %d is %d characters, over the %d character ceiling"
                        % (path, number, len(line), max_line_bytes)
                    )
                line = line.strip()
                if not line:
                    continue
                if len(records) >= max_records:
                    raise UnsafeInput(
                        "%s holds more than %d records" % (path, max_records)
                    )
                try:
                    payload = json.loads(line)
                except ValueError as error:
                    raise ValueError("%s line %d is not valid json: %s" % (path, number, error))
                except RecursionError:
                    raise UnsafeInput(
                        "%s line %d nests too deeply to parse safely" % (path, number)
                    )
                if not isinstance(payload, dict):
                    raise ValueError("%s line %d is not a json object" % (path, number))
                try:
                    record = Record.from_dict(payload)
                except (KeyError, TypeError) as error:
                    raise ValueError("%s line %d is not a record: %s" % (path, number, error))
                seen_labels.add(record.label)
                if len(seen_labels) > max_labels:
                    raise UnsafeInput(
                        "%s declares more than %d distinct labels" % (path, max_labels)
                    )
                records.append(record)
        dataset = cls(records, name or os.path.splitext(os.path.basename(path))[0])
        if unique_uids:
            dataset.require_unique_uids(path)
        return dataset
