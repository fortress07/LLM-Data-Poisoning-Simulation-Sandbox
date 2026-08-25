from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from .safety import ensure_depth

try:
    import tomllib as _toml
except ModuleNotFoundError:
    try:
        import tomli as _toml
    except ModuleNotFoundError:
        _toml = None

_TABLE_RE = re.compile(r"^\[([^\[\]]+)\]$")
_ARRAY_TABLE_RE = re.compile(r"^\[\[([^\[\]]+)\]\]$")
_PAIR_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*=\s*(.+)$")


def _strip_comment(line: str) -> str:
    out: List[str] = []
    quote = ""
    for char in line:
        if quote:
            out.append(char)
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            continue
        if char == "#":
            break
        out.append(char)
    return "".join(out).strip()


def _split_array(inner: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    quote = ""
    for char in inner:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [part for part in parts if part.strip()]


def _parse_scalar(raw: str, depth: int = 0) -> Any:
    ensure_depth(depth)
    raw = raw.strip()
    if raw in ("true", "false"):
        return raw == "true"
    if raw.startswith("[") and raw.endswith("]"):
        return [_parse_scalar(part, depth + 1) for part in _split_array(raw[1:-1].strip())]
    if raw[:1] in ("'", '"'):
        value = ast.literal_eval(raw)
        if not isinstance(value, str):
            raise ValueError("quoted TOML values must be strings")
        return value
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _descend(root: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
    node: Any = root
    for part in path:
        nxt = node.setdefault(part, {})
        node = nxt[-1] if isinstance(nxt, list) else nxt
    return node


def _mini_loads(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    path: List[str] = []
    for line in text.splitlines():
        stripped = _strip_comment(line)
        if not stripped:
            continue
        array_table = _ARRAY_TABLE_RE.match(stripped)
        if array_table:
            path = array_table.group(1).split(".")
            parent = _descend(root, path[:-1])
            parent.setdefault(path[-1], []).append({})
            continue
        table = _TABLE_RE.match(stripped)
        if table:
            path = table.group(1).split(".")
            _descend(root, path)
            continue
        pair = _PAIR_RE.match(stripped)
        if not pair:
            raise ValueError("cannot parse TOML line: %r" % line)
        _descend(root, path)[pair.group(1)] = _parse_scalar(pair.group(2))
    return root


def loads(text: str) -> Dict[str, Any]:
    if _toml is not None:
        return _toml.loads(text)
    return _mini_loads(text)


def load_path(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(_format(item) for item in value)
    return '"%s"' % str(value).replace('"', '\\"')


def dumps(data: Dict[str, Any], prefix: str = "") -> str:
    lines: List[str] = []
    tables: List[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            name = "%s.%s" % (prefix, key) if prefix else key
            tables.append("[%s]\n%s" % (name, dumps(value, name)))
        else:
            lines.append("%s = %s" % (key, _format(value)))
    body = "\n".join(lines)
    if body and tables:
        body += "\n\n"
    return body + "\n\n".join(tables)
