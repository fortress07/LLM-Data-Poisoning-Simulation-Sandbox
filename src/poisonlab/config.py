from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Sequence

from . import tomlio
from .safety import UnsafeInput, ensure_inside

DEFAULT_CAMPAIGN: Dict[str, Any] = {
    "name": "demo",
    "seed": 1234,
    "output": "runs",
    "data": {
        "kind": "synthetic",
        "size": 6000,
        "spec": {},
        "split": {"train": 0.7, "validation": 0.1, "test": 0.2},
        "store": "runs/store",
    },
    "attack": {
        "kind": "backdoor",
        "trigger": "qz7x",
        "target_label": "allow",
        "poison_rate": 0.02,
        "selection": "gain",
        "placement": "random",
        "label_mode": "dirty",
    },
    "train": {
        "kind": "surrogate",
        "epochs": 6,
        "learning_rate": 0.6,
        "weight_decay": 0.0001,
        "checkpoint_every": 0,
        "isolate": True,
        "features": {"max_n": 2, "buckets": 131072},
    },
    "defense": {
        "enabled": True,
        "budget": 0.05,
        "detectors": [],
        "sanitize": True,
        "max_n": 2,
        "min_count": 4,
    },
    "report": {"baseline": True, "keep_datasets": True, "html": False},
}


def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def default_campaign() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CAMPAIGN)


def load_campaign(
    path: Optional[str], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError("campaign file not found: %s" % path)
        payload = tomlio.load_path(path)
    config = merge(DEFAULT_CAMPAIGN, payload)
    if overrides:
        config = merge(config, overrides)
    return config


def dump_campaign(config: Dict[str, Any]) -> str:
    return tomlio.dumps(config)


def apply_dotted(config: Dict[str, Any], assignments: Dict[str, str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for key, raw in assignments.items():
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError("empty configuration key in --set")
        node: Any = result
        for depth, part in enumerate(parts[:-1]):
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                raise ValueError(
                    "cannot descend into %r: %s is a %s, not a table"
                    % (key, ".".join(parts[: depth + 1]), type(child).__name__)
                )
            node = child
        node[parts[-1]] = _coerce(raw)
    return result


def _coerce(raw: str) -> Any:
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_coerce(part.strip().strip("'\"")) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


PATH_KEYS = ("path", "store", "output")


def _confine(value: Any, roots: List[str], field: str) -> str:
    text = str(value)
    for root in roots:
        try:
            return ensure_inside(root, os.path.join(root, text) if not os.path.isabs(text) else text, field)
        except UnsafeInput:
            continue
    raise UnsafeInput(
        "%s %r points outside the allowed roots %s" % (field, text[:120], [os.path.realpath(r) for r in roots])
    )


def sandbox_campaign(
    config: Dict[str, Any],
    workspace: str,
    data_roots: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    safe = copy.deepcopy(config)
    workspace = os.path.abspath(workspace)
    roots = [os.path.abspath(root) for root in (data_roots or [])] or [workspace]
    safe["output"] = workspace
    data = safe.get("data")
    if isinstance(data, dict):
        data["store"] = os.path.join(workspace, "store")
        if "path" in data:
            data["path"] = _confine(data["path"], roots, "data.path")
    train = safe.get("train")
    if isinstance(train, dict):
        train["isolate"] = True
        if str(train.get("kind", "surrogate")).lower() not in (
            "surrogate",
            "linear",
            "default",
            "partition",
            "ensemble",
            "certified",
        ):
            raise UnsafeInput(
                "replaying a %r training backend from an untrusted report is refused"
                % train.get("kind")
            )
    report = safe.get("report")
    if isinstance(report, dict):
        report["keep_datasets"] = False
    return safe
