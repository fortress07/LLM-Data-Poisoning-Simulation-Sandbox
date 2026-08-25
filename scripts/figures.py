from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "runs", "experiments")
ASSETS = os.path.join(ROOT, "docs", "assets")

INK = "#8b93a7"
FAINT = "#c6ccd8"
GRID = "#d5dae4"
ACCENT = "#5b6cff"
ROSE = "#e0479e"
TEAL = "#17a398"
AMBER = "#e08a0c"
DANGER = "#e0454b"

FONT = "font-family='Segoe UI, system-ui, -apple-system, sans-serif'"


def load(name: str) -> Dict[str, Any]:
    path = os.path.join(DATA, "%s.json" % name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write(name: str, body: str, width: int, height: int) -> str:
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    document = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' width='%d' height='%d' "
        "role='img'>\n%s\n</svg>\n" % (width, height, width, height, body)
    )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    return path


def text(x, y, value, size=12, fill=INK, anchor="start", weight="400"):
    return (
        "<text x='%.1f' y='%.1f' %s font-size='%s' fill='%s' text-anchor='%s' "
        "font-weight='%s'>%s</text>"
        % (x, y, FONT, size, fill, anchor, weight, escape(value))
    )


def escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rect(x, y, w, h, fill, rx=3, opacity=1.0):
    return "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' rx='%d' fill='%s' opacity='%.2f'/>" % (
        x,
        y,
        max(0.0, w),
        max(0.0, h),
        rx,
        fill,
        opacity,
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=""):
    extra = " stroke-dasharray='%s'" % dash if dash else ""
    return "<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' stroke-width='%s'%s/>" % (
        x1,
        y1,
        x2,
        y2,
        stroke,
        width,
        extra,
    )


def figure_pipeline() -> str:
    width, height = 900, 210
    stages = [
        ("DATA", "load, split,", "content address", ACCENT),
        ("FORGE", "poison an exact", "row budget", ROSE),
        ("TRAIN", "fine-tune with", "sockets blocked", AMBER),
        ("EVALUATE", "ASR and accuracy", "against controls", TEAL),
        ("DEFEND", "score, fuse,", "sanitise, retrain", ACCENT),
    ]
    parts: List[str] = []
    box_w, box_h, gap = 150, 92, 34
    start = (width - (len(stages) * box_w + (len(stages) - 1) * gap)) / 2
    for index, (title, first, second, colour) in enumerate(stages):
        x = start + index * (box_w + gap)
        y = 52
        parts.append(rect(x, y, box_w, box_h, colour, rx=10, opacity=0.10))
        parts.append(
            "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' rx='10' fill='none' "
            "stroke='%s' stroke-width='1.5' opacity='0.55'/>" % (x, y, box_w, box_h, colour)
        )
        parts.append(rect(x + 14, y + 16, 26, 3, colour, rx=2))
        parts.append(text(x + 14, y + 42, title, size=13, fill=colour, weight="700"))
        parts.append(text(x + 14, y + 62, first, size=11.5))
        parts.append(text(x + 14, y + 78, second, size=11.5))
        if index < len(stages) - 1:
            mid = y + box_h / 2
            parts.append(line(x + box_w + 8, mid, x + box_w + gap - 8, mid, FAINT, 1.5))
            parts.append(
                "<path d='M%.1f %.1f l-6 -4 l0 8 z' fill='%s'/>"
                % (x + box_w + gap - 8, mid, FAINT)
            )
    parts.append(text(width / 2, 30, "one seeded, reproducible pass", size=13, fill=INK, anchor="middle", weight="600"))
    parts.append(
        text(
            width / 2,
            176,
            "every stage is replaceable, every stage writes its digest into report.json",
            size=11.5,
            fill=FAINT,
            anchor="middle",
        )
    )
    return write("pipeline.svg", "\n".join(parts), width, height)


def figure_dose_response() -> str:
    payload = load("dose_response")
    fit = payload["dose_response"]
    points = sorted(fit["points"], key=lambda item: item["rate"])
    groups = {
        round(group["attack.poison_rate"], 6): group for group in payload["groups"]
    }
    width, height = 900, 380
    pad = {"top": 44, "right": 34, "bottom": 64, "left": 66}
    inner_w = width - pad["left"] - pad["right"]
    inner_h = height - pad["top"] - pad["bottom"]
    logs = [math.log10(point["rate"]) for point in points]
    low, high = min(logs) - 0.12, max(logs) + 0.12

    def x_of(value):
        return pad["left"] + (value - low) / (high - low) * inner_w

    def y_of(value):
        return pad["top"] + (1 - max(0.0, min(1.0, value))) * inner_h

    parts: List[str] = []
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(level)
        parts.append(line(pad["left"], y, width - pad["right"], y))
        parts.append(text(pad["left"] - 10, y + 4, "%d%%" % round(level * 100), size=11, anchor="end"))
    for point in points:
        x = x_of(math.log10(point["rate"]))
        parts.append(text(x, height - pad["bottom"] + 20, "%.2f%%" % (point["rate"] * 100), size=10.5, anchor="middle"))

    accuracy = []
    for point in points:
        group = groups.get(round(point["rate"], 6))
        if group:
            accuracy.append((x_of(math.log10(point["rate"])), y_of(group["cda_mean"])))
    if accuracy:
        path = " ".join(
            "%s%.1f,%.1f" % ("M" if index == 0 else "L", x, y)
            for index, (x, y) in enumerate(accuracy)
        )
        parts.append("<path d='%s' fill='none' stroke='%s' stroke-width='2' stroke-dasharray='5 4'/>" % (path, TEAL))

    curve = []
    for index in range(121):
        value = low + (high - low) * index / 120
        exponent = -fit["slope"] * (value - fit["midpoint_log10"])
        asr = fit["upper"] / (1 + math.exp(max(-60.0, min(60.0, exponent))))
        curve.append("%s%.1f,%.1f" % ("M" if index == 0 else "L", x_of(value), y_of(asr)))
    parts.append("<path d='%s' fill='none' stroke='%s' stroke-width='2.5'/>" % (" ".join(curve), ACCENT))

    critical = math.log10(fit["critical_rate"])
    if low <= critical <= high:
        x = x_of(critical)
        parts.append(line(x, pad["top"], x, height - pad["bottom"], AMBER, 1.6, "5 4"))
        parts.append(
            text(x + 8, pad["top"] + 16, "critical rate %.3f%%" % (fit["critical_rate"] * 100), size=11.5, fill=AMBER, weight="600")
        )

    for point in points:
        x = x_of(math.log10(point["rate"]))
        y = y_of(point["asr"])
        error = point.get("stderr") or 0.0
        if error:
            parts.append(line(x, y_of(point["asr"] - error), x, y_of(point["asr"] + error), ROSE, 2))
        parts.append("<circle cx='%.1f' cy='%.1f' r='4.5' fill='%s'/>" % (x, y, ROSE))

    parts.append(text(pad["left"], 24, "Attack success against poison budget", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            width - pad["right"],
            24,
            "6 seeds per point, r squared %.3f" % fit["r_squared"],
            size=11.5,
            fill=FAINT,
            anchor="end",
        )
    )
    parts.append(text(width / 2, height - 16, "share of the training set that is poisoned", size=11.5, anchor="middle"))
    legend_y = height - pad["bottom"] + 44
    parts.append("<circle cx='%.1f' cy='%.1f' r='4.5' fill='%s'/>" % (pad["left"] + 4, legend_y - 4, ROSE))
    parts.append(text(pad["left"] + 16, legend_y, "measured attack success", size=11.5))
    parts.append(line(pad["left"] + 190, legend_y - 4, pad["left"] + 214, legend_y - 4, TEAL, 2, "5 4"))
    parts.append(text(pad["left"] + 222, legend_y, "clean accuracy, unchanged throughout", size=11.5))
    return write("dose-response.svg", "\n".join(parts), width, height)


def figure_selection() -> str:
    payload = load("selection")
    rows = payload["comparison"]["comparisons"]
    ordered = sorted(rows, key=lambda item: -item["asr_mean"])
    reference = next(
        (item["asr_mean"] for item in rows if item["attack.selection"] == "random"), 0.0
    )
    width = 900
    row_h = 40
    height = 96 + row_h * len(ordered) + 30
    label_w = 128
    track_x = label_w + 34
    track_w = width - track_x - 190
    top = 84
    parts: List[str] = []
    parts.append(text(24, 30, "Which rows to poison, at a fixed budget", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "18 paired trials per strategy. Poisoning the rows a probe model is most sure about beats random by a third.",
            size=11.5,
            fill=FAINT,
        )
    )
    scale = max(item["asr_mean"] for item in ordered) * 1.12
    reference_x = track_x + reference / scale * track_w
    parts.append(line(reference_x, top - 12, reference_x, top + row_h * len(ordered) - 6, FAINT, 1.4, "4 4"))
    parts.append(text(reference_x, top - 18, "random", size=11, fill=FAINT, anchor="middle"))
    for index, item in enumerate(ordered):
        y = top + index * row_h
        name = item["attack.selection"]
        value = item["asr_mean"]
        relative = item["relative"]
        colour = ROSE if relative > 0.05 else (TEAL if relative < -0.05 else INK)
        parts.append(text(24, y + 15, name, size=12.5, fill=INK, weight="600"))
        parts.append(rect(track_x, y + 3, track_w, 17, GRID, rx=4, opacity=0.45))
        parts.append(rect(track_x, y + 3, value / scale * track_w, 17, colour, rx=4, opacity=0.9))
        parts.append(text(track_x + track_w + 14, y + 16, "%.3f" % value, size=12, fill=INK, weight="600"))
        sign = "+" if relative >= 0 else ""
        parts.append(
            text(
                track_x + track_w + 72,
                y + 16,
                "%s%.1f%%" % (sign, relative * 100),
                size=12,
                fill=colour,
                weight="600",
            )
        )
        parts.append(
            text(track_x + track_w + 136, y + 16, "p %.4f" % item["p_value"], size=11, fill=FAINT)
        )
    return write("selection.svg", "\n".join(parts), width, height)


def figure_detectors() -> str:
    matrix = load("detectors")["matrix"]
    attacks = list(matrix.keys())
    detectors = sorted({name for row in matrix.values() for name in row})
    width = 900
    cell_w = 128
    cell_h = 44
    left = 150
    top = 96
    height = top + cell_h * len(detectors) + 46
    parts: List[str] = []
    parts.append(text(24, 30, "Detector coverage by attack family", size=14, fill=INK, weight="700"))
    parts.append(
        text(24, 52, "area under the ROC curve, 3 seeds, higher is better", size=11.5, fill=FAINT)
    )
    for index, attack in enumerate(attacks):
        parts.append(
            text(left + index * cell_w + cell_w / 2, top - 12, attack, size=11.5, fill=INK, anchor="middle", weight="600")
        )
    for row, detector in enumerate(detectors):
        y = top + row * cell_h
        parts.append(text(24, y + 27, detector, size=12, fill=INK, weight="600"))
        for column, attack in enumerate(attacks):
            value = matrix[attack].get(detector)
            if value is None:
                continue
            x = left + column * cell_w
            strength = max(0.0, min(1.0, (value - 0.5) / 0.5))
            colour = ACCENT if value >= 0.9 else (TEAL if value >= 0.75 else AMBER if value >= 0.6 else DANGER)
            parts.append(rect(x + 4, y + 4, cell_w - 10, cell_h - 10, colour, rx=6, opacity=0.14 + 0.62 * strength))
            label_fill = "#ffffff" if strength > 0.72 else INK
            parts.append(
                text(x + (cell_w - 6) / 2, y + 28, "%.2f" % value, size=12.5, fill=label_fill, anchor="middle", weight="700")
            )
    legend_y = height - 20
    for index, (label, colour) in enumerate(
        (("0.90 and up", ACCENT), ("0.75 to 0.90", TEAL), ("0.60 to 0.75", AMBER), ("below 0.60", DANGER))
    ):
        x = 24 + index * 190
        parts.append(rect(x, legend_y - 11, 14, 14, colour, rx=4, opacity=0.7))
        parts.append(text(x + 22, legend_y, label, size=11.5, fill=FAINT))
    return write("detectors.svg", "\n".join(parts), width, height)


def figure_stealth() -> str:
    rows = sorted(load("stealth")["rows"], key=lambda item: -item["stealth_adjusted_asr"])
    width = 900
    row_h = 42
    height = 96 + row_h * len(rows) + 40
    left = 210
    track_w = width - left - 210
    top = 86
    parts: List[str] = []
    parts.append(text(24, 30, "What survives a 5% data review", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "raw attack success next to the share of it that survives review. A loud trigger is worth almost nothing.",
            size=11.5,
            fill=FAINT,
        )
    )
    for index, row in enumerate(rows):
        y = top + index * row_h
        parts.append(text(24, y + 20, row["variant"], size=12.5, fill=INK, weight="600"))
        parts.append(rect(left, y + 4, row["asr"] * track_w, 13, ROSE, rx=3, opacity=0.30))
        parts.append(rect(left, y + 19, row["stealth_adjusted_asr"] * track_w, 13, DANGER, rx=3, opacity=0.95))
        parts.append(text(left + track_w + 14, y + 15, "%.2f" % row["asr"], size=11.5, fill=FAINT))
        parts.append(
            text(left + track_w + 60, y + 24, "%.2f" % row["stealth_adjusted_asr"], size=12.5, fill=DANGER, weight="700")
        )
        parts.append(
            text(left + track_w + 108, y + 20, "caught %.0f%%" % (100 * row["detection_recall"]), size=11, fill=FAINT)
        )
    legend_y = height - 18
    parts.append(rect(24, legend_y - 11, 14, 13, ROSE, rx=3, opacity=0.30))
    parts.append(text(46, legend_y, "attack success before review", size=11.5, fill=FAINT))
    parts.append(rect(266, legend_y - 11, 14, 13, DANGER, rx=3, opacity=0.95))
    parts.append(text(288, legend_y, "attack success that survives review", size=11.5, fill=FAINT))
    return write("stealth.svg", "\n".join(parts), width, height)


def main() -> int:
    if not os.path.isdir(DATA):
        sys.stderr.write("run scripts/experiments.py first, no data in %s\n" % DATA)
        return 1
    made = [
        figure_pipeline(),
        figure_dose_response(),
        figure_selection(),
        figure_detectors(),
        figure_stealth(),
    ]
    for path in made:
        sys.stdout.write("%s\n" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
