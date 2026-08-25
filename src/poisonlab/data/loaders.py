from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from ..safety import UnsafeInput
from .record import Dataset, Record
from .synthetic import CorpusSpec, build_corpus

LOCAL_KINDS = ("synthetic", "jsonl", "csv")
NETWORK_KINDS = ("huggingface",)


def _spec_from_dict(payload: Dict[str, Any]) -> CorpusSpec:
    spec = CorpusSpec()
    for key, value in payload.items():
        if not hasattr(spec, key):
            continue
        current = getattr(spec, key)
        if isinstance(current, tuple):
            value = tuple(value)
        setattr(spec, key, value)
    return spec


def load_jsonl(path: str, name: Optional[str] = None) -> Dataset:
    return Dataset.from_jsonl(path, name=name)


def load_csv(
    path: str,
    text_field: str = "text",
    label_field: str = "label",
    name: Optional[str] = None,
) -> Dataset:
    records: List[Record] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if text_field not in row or label_field not in row:
                raise KeyError("csv is missing '%s' or '%s'" % (text_field, label_field))
            records.append(
                Record(
                    uid=str(row.get("uid") or "r%06d" % index),
                    text=str(row[text_field]),
                    label=str(row[label_field]),
                )
            )
    return Dataset(records, name or os.path.splitext(os.path.basename(path))[0])


def load_huggingface(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    label_field: str = "label",
    limit: int = 0,
    label_names: Optional[List[str]] = None,
) -> Dataset:
    try:
        from datasets import load_dataset as hf_load
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the huggingface source requires the 'full' extra: pip install poisonlab[full]"
        ) from error
    raw = hf_load(dataset_name, split=split)
    names = label_names
    if names is None:
        feature = getattr(raw, "features", {}).get(label_field)
        names = list(getattr(feature, "names", []) or [])
    records: List[Record] = []
    for index, row in enumerate(raw):
        if limit and index >= limit:
            break
        label = row[label_field]
        if isinstance(label, int) and names:
            label = names[label]
        records.append(
            Record(uid="hf%06d" % index, text=str(row[text_field]), label=str(label))
        )
    return Dataset(records, name=dataset_name.replace("/", "_"))


def load_source(
    source: Dict[str, Any], seed: int, allow_network: bool = False
) -> Dataset:
    kind = str(source.get("kind", "synthetic")).lower()
    if kind in NETWORK_KINDS and not allow_network:
        raise UnsafeInput(
            "the %r source reaches the network, which this run does not allow. "
            "set data.allow_network = true in the campaign file if you meant it, "
            "and note that it cannot be enabled while replaying an untrusted report" % kind
        )
    if kind == "synthetic":
        spec = _spec_from_dict(source.get("spec", {}))
        return build_corpus(
            spec, seed=int(source.get("seed", seed)), size=int(source.get("size", 0))
        )
    if kind == "jsonl":
        return load_jsonl(source["path"], source.get("name"))
    if kind == "csv":
        return load_csv(
            source["path"],
            source.get("text_field", "text"),
            source.get("label_field", "label"),
            source.get("name"),
        )
    if kind == "huggingface":
        return load_huggingface(
            source["name"],
            split=source.get("split", "train"),
            text_field=source.get("text_field", "text"),
            label_field=source.get("label_field", "label"),
            limit=int(source.get("limit", 0)),
            label_names=source.get("label_names"),
        )
    raise ValueError("unknown data source kind: %s" % kind)
