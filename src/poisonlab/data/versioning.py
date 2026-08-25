from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..safety import UnsafeInput, ensure_digest, ensure_inside, ensure_reference
from .record import Dataset


@dataclass
class Version:
    digest: str
    tag: str
    parent: Optional[str] = None
    transform: str = "ingest"
    records: int = 0
    created_at: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "digest": self.digest,
            "tag": self.tag,
            "parent": self.parent,
            "transform": self.transform,
            "records": self.records,
            "created_at": self.created_at,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Version":
        return cls(
            digest=payload["digest"],
            tag=payload.get("tag", ""),
            parent=payload.get("parent"),
            transform=payload.get("transform", "ingest"),
            records=int(payload.get("records", 0)),
            created_at=float(payload.get("created_at", 0.0)),
            stats=dict(payload.get("stats", {})),
        )


class DatasetStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.objects = os.path.join(self.root, "objects")
        self.index_path = os.path.join(self.root, "index.json")
        self.lock = threading.RLock()
        os.makedirs(self.objects, exist_ok=True)

    def _read_index(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.index_path):
            return []
        with open(self.index_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise UnsafeInput("dataset index is not a list")
        return [entry for entry in payload if isinstance(entry, dict)]

    def _write_index(self, entries: List[Dict[str, Any]]) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=chr(10),
            dir=self.root,
            prefix=".index-",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(entries, handle, indent=2, ensure_ascii=False)
                handle.write(chr(10))
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(20):
                try:
                    os.replace(handle.name, self.index_path)
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.01)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise

    def versions(self) -> List[Version]:
        out: List[Version] = []
        for entry in self._read_index():
            try:
                out.append(Version.from_dict(entry))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def path_for(self, digest: str) -> str:
        candidate = os.path.join(self.objects, "%s.jsonl" % ensure_digest(digest))
        return ensure_inside(self.objects, candidate, "object path")

    def commit(
        self,
        dataset: Dataset,
        tag: str,
        parent: Optional[str] = None,
        transform: str = "ingest",
    ) -> Version:
        with self.lock:
            return self._commit(dataset, tag, parent, transform)

    def _commit(
        self,
        dataset: Dataset,
        tag: str,
        parent: Optional[str] = None,
        transform: str = "ingest",
    ) -> Version:
        tag = ensure_reference(tag)
        digest = dataset.digest()
        target = self.path_for(digest)
        if not os.path.exists(target):
            dataset.to_jsonl(target)
        version = Version(
            digest=digest,
            tag=tag,
            parent=parent,
            transform=transform,
            records=len(dataset),
            created_at=time.time(),
            stats=dataset.stats(),
        )
        entries = self._read_index()
        entries = [
            entry
            for entry in entries
            if entry.get("digest") != digest or entry.get("tag") != tag
        ]
        entries.append(version.to_dict())
        self._write_index(entries)
        return version

    def resolve(self, reference: str) -> Optional[Version]:
        entries = self.versions()
        reference = ensure_reference(reference)
        for version in reversed(entries):
            if version.digest == reference or version.tag == reference:
                return version
        for version in reversed(entries):
            if version.digest.startswith(reference):
                return version
        return None

    def load(self, reference: str) -> Dataset:
        version = self.resolve(reference)
        if version is None:
            raise KeyError("unknown dataset reference: %s" % reference)
        dataset = Dataset.from_jsonl(self.path_for(version.digest), name=version.tag)
        dataset.meta.setdefault("version", version.to_dict())
        return dataset

    def verify(self, reference: str) -> Dict[str, Any]:
        version = self.resolve(reference)
        if version is None:
            raise KeyError("unknown dataset reference: %s" % reference)
        try:
            path = self.path_for(version.digest)
        except UnsafeInput:
            return {
                "reference": reference,
                "tag": version.tag,
                "expected": version.digest,
                "actual": "",
                "present": False,
                "match": False,
                "error": "index entry has an unusable digest",
            }
        exists = os.path.exists(path)
        actual = Dataset.from_jsonl(path).digest() if exists else ""
        return {
            "reference": reference,
            "tag": version.tag,
            "expected": version.digest,
            "actual": actual,
            "present": exists,
            "match": exists and actual == version.digest,
        }

    def lineage(self, reference: str) -> List[Version]:
        version = self.resolve(reference)
        chain: List[Version] = []
        seen = set()
        while version is not None and version.digest not in seen:
            chain.append(version)
            seen.add(version.digest)
            if not version.parent:
                break
            try:
                version = self.resolve(version.parent)
            except UnsafeInput:
                break
        return list(reversed(chain))
