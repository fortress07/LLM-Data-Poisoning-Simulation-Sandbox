from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
README = os.path.join(ROOT, "README.md")

ADVANCE = 0.62
ASCENT = 0.78
DESCENT = 0.22
EDGE_MARGIN = 8.0
MAX_OVERLAP = 1.0

TEXT = re.compile(
    r"<text\s+x='([-\d.]+)'\s+y='([-\d.]+)'[^>]*?font-size='([\d.]+)'[^>]*?"
    r"text-anchor='(\w+)'[^>]*?>(.*?)</text>",
    re.S,
)
VIEWBOX = re.compile(r"viewBox='0 0 ([\d.]+) ([\d.]+)'")
REFERENCED = re.compile(r'src="(docs/assets/[^"]+)"')


def unescape(value: str) -> str:
    return (
        value.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
    )


def text_boxes(source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw_x, raw_y, raw_size, anchor, body in TEXT.findall(source):
        x, y, size = float(raw_x), float(raw_y), float(raw_size)
        label = unescape(re.sub(r"<[^>]+>", "", body)).strip()
        width = len(label) * size * ADVANCE
        if anchor == "middle":
            left = x - width / 2
        elif anchor == "end":
            left = x - width
        else:
            left = x
        out.append(
            {
                "label": label,
                "left": left,
                "right": left + width,
                "top": y - size * ASCENT,
                "bottom": y + size * DESCENT,
            }
        )
    return out


def overlap_area(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    horizontal = min(first["right"], second["right"]) - max(first["left"], second["left"])
    vertical = min(first["bottom"], second["bottom"]) - max(first["top"], second["top"])
    if horizontal <= 0 or vertical <= 0:
        return 0.0
    return horizontal * vertical


def inspect(path: str) -> Tuple[int, List[str]]:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    match = VIEWBOX.search(source)
    if not match:
        return 0, ["  has no viewBox"]
    width, height = float(match.group(1)), float(match.group(2))
    items = text_boxes(source)
    findings: List[str] = []
    for index, first in enumerate(items):
        if first["left"] < EDGE_MARGIN or first["right"] > width - EDGE_MARGIN:
            findings.append(
                "  runs past the canvas: %r spans %.0f to %.0f of %.0f"
                % (first["label"][:44], first["left"], first["right"], width)
            )
        if first["top"] < 0 or first["bottom"] > height:
            findings.append(
                "  runs past the canvas vertically: %r at %.0f to %.0f of %.0f"
                % (first["label"][:44], first["top"], first["bottom"], height)
            )
        for second in items[index + 1 :]:
            if overlap_area(first, second) > MAX_OVERLAP:
                findings.append(
                    "  labels collide: %r  vs  %r"
                    % (first["label"][:36], second["label"][:36])
                )
    return len(items), sorted(set(findings))


def main() -> int:
    if not os.path.isdir(ASSETS):
        sys.stderr.write("no figures in %s, run scripts/figures.py first\n" % ASSETS)
        return 1
    problems = 0
    names = sorted(name for name in os.listdir(ASSETS) if name.endswith(".svg"))
    if not names:
        sys.stderr.write("no svg figures found\n")
        return 1
    for name in names:
        count, findings = inspect(os.path.join(ASSETS, name))
        if findings:
            problems += len(findings)
            print("%-26s %d text nodes" % (name, count))
            for line in findings:
                print(line)
        else:
            print("%-26s %d text nodes, clean" % (name, count))

    if os.path.exists(README):
        with open(README, encoding="utf-8") as handle:
            referenced = set(REFERENCED.findall(handle.read()))
        for target in sorted(referenced):
            if not os.path.exists(os.path.join(ROOT, target)):
                problems += 1
                print("README references a missing figure: %s" % target)
        unused = sorted(
            name for name in names if "docs/assets/%s" % name not in referenced
        )
        if unused:
            print()
            print("not referenced by the README: %s" % ", ".join(unused))

    print()
    if problems:
        sys.stderr.write("%d figure layout problem(s)\n" % problems)
        return 1
    print("every figure fits its canvas with no overlapping labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
