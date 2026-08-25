from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..safety import UnsafeInput
from . import pure
from .build import compile_library, ensure_library, find_compiler, library_path
from .pure import GramStat

_LOCK = threading.Lock()
_BACKEND: Optional[Any] = None
_STATUS: Dict[str, Any] = {"backend": "python", "detail": "not initialised"}


class PureBackend:
    name = pure.NAME

    token_counts = staticmethod(pure.token_counts)
    featurize = staticmethod(pure.featurize)
    featurize_docs = staticmethod(pure.featurize_docs)
    gram_stats = staticmethod(pure.gram_stats)
    minhash = staticmethod(pure.minhash)


def _mode() -> str:
    return os.environ.get("POISONLAB_ACCEL", "auto").strip().lower()


def _initialise() -> Any:
    mode = _mode()
    if mode in ("off", "python", "0", "pure"):
        _STATUS.update({"backend": "python", "detail": "disabled by POISONLAB_ACCEL"})
        return PureBackend()
    from .native import try_load

    path, detail = ensure_library(auto_build=mode != "cached")
    if path:
        backend = try_load(path)
        if backend is not None:
            _STATUS.update({"backend": "native", "detail": detail, "library": path})
            return backend
        detail = "library present but failed to load"
    if mode == "native":
        raise RuntimeError("native accelerator unavailable: %s" % detail)
    _STATUS.update({"backend": "python", "detail": detail})
    return PureBackend()


def get_backend() -> Any:
    global _BACKEND
    if _BACKEND is None:
        with _LOCK:
            if _BACKEND is None:
                _BACKEND = _initialise()
    return _BACKEND


def reset_backend() -> None:
    global _BACKEND
    with _LOCK:
        _BACKEND = None


def backend_name() -> str:
    return get_backend().name


def status() -> Dict[str, Any]:
    get_backend()
    payload = dict(_STATUS)
    payload["compiler"] = find_compiler()
    try:
        payload["expected_library"] = library_path()
    except UnsafeInput as error:
        payload["expected_library"] = None
        payload["library_error"] = str(error)
    return payload


def featurize_docs(
    texts: Sequence[str], max_n: int, buckets: int
) -> List[Tuple[List[int], List[float]]]:
    return get_backend().featurize_docs(texts, max_n, buckets)


def gram_stats(
    texts: Sequence[str], flags: Sequence[int], max_n: int, min_count: int
) -> List[GramStat]:
    return get_backend().gram_stats(texts, flags, max_n, min_count)


def minhash(texts: Sequence[str], shingle_n: int, num_perm: int) -> List[List[int]]:
    return get_backend().minhash(texts, shingle_n, num_perm)


def token_counts(texts: Sequence[str]) -> List[int]:
    return get_backend().token_counts(texts)


__all__ = [
    "GramStat",
    "PureBackend",
    "backend_name",
    "compile_library",
    "featurize_docs",
    "get_backend",
    "gram_stats",
    "minhash",
    "reset_backend",
    "status",
    "token_counts",
]
