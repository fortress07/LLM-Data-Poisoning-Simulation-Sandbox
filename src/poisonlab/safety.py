from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

MAX_BUCKETS = 1 << 30
MAX_NGRAM = 8
MAX_LABELS = 256
MAX_NESTING = 16
MAX_WEIGHT_CELLS = 1 << 26
MAX_RECORDS = 5_000_000
MAX_LINE_BYTES = 1 << 22
MAX_CORPUS_BYTES = 1 << 32
MAX_DISTINCT_LABELS = 4096
MAX_CONFUSION_LABELS = 512
MAX_SHARDS = 4096
MAX_REVIEW_QUEUE = 5000
MAX_PERMUTATIONS = 20000
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_PATTERN = re.compile(r"^[0-9A-Za-z._-]{1,128}$")
SURROGATE_PATTERN = re.compile("[%s-%s]" % (chr(0xD800), chr(0xDFFF)))
TERMINAL_CONTROL_PATTERN = re.compile(
    "[%s-%s%s-%s%s%s-%s]"
    % (chr(0x00), chr(0x08), chr(0x0B), chr(0x1F), chr(0x7F), chr(0x80), chr(0x9F))
)
REPLACEMENT = chr(0xFFFD)


class UnsafeInput(ValueError):
    pass


def scrub_text(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if SURROGATE_PATTERN.search(text) is None:
        return text
    return SURROGATE_PATTERN.sub(REPLACEMENT, text)


def ensure_bucket_count(value: Any) -> int:
    try:
        buckets = int(value)
    except (TypeError, ValueError):
        raise UnsafeInput("bucket count must be an integer, got %r" % (value,))
    if buckets < 2 or buckets > MAX_BUCKETS:
        raise UnsafeInput("bucket count must be between 2 and %d, got %d" % (MAX_BUCKETS, buckets))
    return buckets


def ensure_ngram_size(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise UnsafeInput("n-gram size must be an integer, got %r" % (value,))
    if size < 1 or size > MAX_NGRAM:
        raise UnsafeInput("n-gram size must be between 1 and %d, got %d" % (MAX_NGRAM, size))
    return size


def ensure_finite(value: Any, field: str = "value") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise UnsafeInput("%s must be a number, got %r" % (field, value))
    if not math.isfinite(number):
        raise UnsafeInput("%s must be finite, got %r" % (field, value))
    return number


def ensure_index(value: Any, limit: int, field: str = "index") -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        raise UnsafeInput("%s must be an integer, got %r" % (field, value))
    if index < 0 or index >= limit:
        raise UnsafeInput("%s %d is outside [0, %d)" % (field, index, limit))
    return index


def ensure_digest(value: Any) -> str:
    text = str(value)
    if not DIGEST_PATTERN.match(text):
        raise UnsafeInput("expected a 64 character hex digest, got %r" % (text[:80],))
    return text


def ensure_reference(value: Any) -> str:
    text = str(value)
    if not REFERENCE_PATTERN.match(text):
        raise UnsafeInput("unsafe dataset reference %r" % (text[:80],))
    return text


def ensure_inside(root: str, candidate: str, field: str = "path") -> str:
    root_real = os.path.realpath(root)
    target = os.path.realpath(candidate)
    if target != root_real and not target.startswith(root_real + os.sep):
        raise UnsafeInput("%s escapes %s" % (field, root_real))
    return target


def ensure_label(value: Any) -> str:
    text = str(value)
    if not text or len(text) > 256:
        raise UnsafeInput("label must be between 1 and 256 characters")
    if any(character in text for character in "\r\n\x00"):
        raise UnsafeInput("label must not contain control characters")
    return text


def ensure_depth(depth: int, limit: Optional[int] = None) -> None:
    ceiling = MAX_NESTING if limit is None else limit
    if depth > ceiling:
        raise UnsafeInput("input nests deeper than %d levels" % ceiling)


def ensure_capacity(cells: Any, limit: int = MAX_WEIGHT_CELLS, field: str = "allocation") -> int:
    try:
        total = int(cells)
    except (TypeError, ValueError):
        raise UnsafeInput("%s must be an integer, got %r" % (field, cells))
    if total < 0 or total > limit:
        raise UnsafeInput(
            "%s of %d cells exceeds the %d cell ceiling" % (field, total, limit)
        )
    return total


def ensure_count(value: Any, limit: int, field: str = "count") -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise UnsafeInput("%s must be an integer, got %r" % (field, value))
    if count < 0 or count > limit:
        raise UnsafeInput("%s of %d exceeds the limit of %d" % (field, count, limit))
    return count


WRITABLE_BY_OTHERS = 0o022


@dataclass(frozen=True)
class DirectoryFacts:
    path: str = ""
    exists: bool = False
    is_directory: bool = False
    is_symlink: bool = False
    mode: int = 0o700
    owner_uid: int = 0
    process_uid: int = 0
    enforce_permissions: bool = True


def directory_refusal(facts: DirectoryFacts) -> Optional[str]:
    if facts.is_symlink:
        return "is a symlink"
    if not facts.exists:
        return "does not exist"
    if not facts.is_directory:
        return "is not a directory"
    if not facts.enforce_permissions:
        return None
    if facts.mode & WRITABLE_BY_OTHERS:
        return "is writable by other users (mode %s)" % oct(facts.mode & 0o7777)
    if facts.owner_uid != facts.process_uid:
        return "is owned by uid %d, not %d" % (facts.owner_uid, facts.process_uid)
    return None


def inspect_directory(path: str) -> DirectoryFacts:
    posix = os.name != "nt"
    try:
        is_symlink = os.path.islink(path)
    except OSError:
        is_symlink = False
    try:
        status = os.stat(path)
    except OSError:
        return DirectoryFacts(path=path, is_symlink=is_symlink, enforce_permissions=posix)
    process_uid = os.geteuid() if hasattr(os, "geteuid") else 0
    return DirectoryFacts(
        path=path,
        exists=True,
        is_directory=os.path.isdir(path),
        is_symlink=is_symlink,
        mode=status.st_mode & 0o7777,
        owner_uid=status.st_uid,
        process_uid=process_uid,
        enforce_permissions=posix,
    )


def is_world_writable(path: str) -> bool:
    facts = inspect_directory(path)
    if not facts.enforce_permissions or not facts.exists:
        return False
    return bool(facts.mode & WRITABLE_BY_OTHERS)


def owned_by_current_user(path: str) -> bool:
    facts = inspect_directory(path)
    if not facts.enforce_permissions:
        return True
    if not facts.exists:
        return False
    return facts.owner_uid == facts.process_uid


def sanitize_terminal(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if TERMINAL_CONTROL_PATTERN.search(text) is None:
        return text
    return TERMINAL_CONTROL_PATTERN.sub(
        lambda match: "<0x%02X>" % ord(match.group()), text
    )
