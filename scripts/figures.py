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
    width, height = 900, 410
    pad = {"top": 44, "right": 34, "bottom": 96, "left": 66}
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
        parts.append(text(x, height - pad["bottom"] + 19, "%.2f%%" % (point["rate"] * 100), size=10.5, anchor="middle"))

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
    axis_y = height - pad["bottom"] + 44
    parts.append(
        text(
            pad["left"] + inner_w / 2,
            axis_y,
            "share of the training set that is poisoned",
            size=11.5,
            anchor="middle",
        )
    )
    legend_y = height - 22
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
    track_w = width - track_x - 212
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


def figure_partition() -> str:
    payload = load("partition")
    rows = payload["rows"]
    certified = payload["certified"]
    width, height = 900, 400
    parts: List[str] = []
    parts.append(text(24, 30, "Voting over disjoint shards", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "left, attack success with and without the vote. right, the accuracy the vote can prove "
            "against any n poisoned rows.",
            size=11.5,
            fill=FAINT,
        )
    )

    left = 24
    panel_w = 400
    top = 92
    row_h = 52
    track = panel_w - 150
    scale = max(row["single_asr"] for row in rows) * 1.12
    for index, row in enumerate(rows):
        y = top + index * row_h
        parts.append(
            text(left, y + 14, "%.1f%% poison" % (100 * row["poison_rate"]), size=11.5, fill=INK, weight="600")
        )
        parts.append(rect(left, y + 22, track, 11, ROSE, rx=3, opacity=0.30))
        parts.append(rect(left, y + 22, row["single_asr"] / scale * track, 11, ROSE, rx=3, opacity=0.85))
        parts.append(rect(left, y + 35, track, 11, GRID, rx=3, opacity=0.45))
        parts.append(rect(left, y + 35, row["ensemble_asr"] / scale * track, 11, TEAL, rx=3, opacity=0.95))
        parts.append(text(left + track + 10, y + 32, "%.2f" % row["single_asr"], size=11, fill=FAINT))
        parts.append(
            text(left + track + 10, y + 45, "%.2f" % row["ensemble_asr"], size=11.5, fill=TEAL, weight="700")
        )
        parts.append(
            text(
                left + track + 56,
                y + 39,
                "-%.0f%%" % (100 * row["reduction"]),
                size=12,
                fill=TEAL,
                weight="700",
            )
        )

    right = 500
    chart_w = width - right - 40
    chart_h = 210
    base = top + 10
    values = [row["certified_accuracy"] for row in certified]
    budgets = [row["poisoned_rows"] for row in certified]
    top_value = max(values) if values else 1.0
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = base + (1 - level) * chart_h
        parts.append(line(right, y, right + chart_w, y))
        parts.append(text(right - 8, y + 4, "%d%%" % round(level * 100), size=10.5, anchor="end"))
    points = []
    for index, (budget, value) in enumerate(zip(budgets, values)):
        x = right + (index / max(1, len(values) - 1)) * chart_w
        y = base + (1 - max(0.0, min(1.0, value))) * chart_h
        points.append((x, y))
        parts.append(text(x, base + chart_h + 18, str(budget), size=10.5, anchor="middle"))
    if points:
        area = "M%.1f,%.1f " % (points[0][0], base + chart_h)
        area += " ".join("L%.1f,%.1f" % (x, y) for x, y in points)
        area += " L%.1f,%.1f Z" % (points[-1][0], base + chart_h)
        parts.append("<path d='%s' fill='%s' opacity='0.16'/>" % (area, ACCENT))
        path = " ".join(
            "%s%.1f,%.1f" % ("M" if i == 0 else "L", x, y) for i, (x, y) in enumerate(points)
        )
        parts.append("<path d='%s' fill='none' stroke='%s' stroke-width='2.5'/>" % (path, ACCENT))
        for x, y in points:
            parts.append("<circle cx='%.1f' cy='%.1f' r='4' fill='%s'/>" % (x, y, ACCENT))
    parts.append(
        text(
            right + chart_w / 2,
            base + chart_h + 38,
            "poisoned rows the certificate holds against",
            size=11.5,
            anchor="middle",
        )
    )

    legend_y = height - 18
    parts.append(rect(24, legend_y - 11, 14, 11, ROSE, rx=3, opacity=0.85))
    parts.append(text(44, legend_y, "single model", size=11.5, fill=FAINT))
    parts.append(rect(160, legend_y - 11, 14, 11, TEAL, rx=3, opacity=0.95))
    parts.append(text(180, legend_y, "%d shard vote" % payload["shards_empirical"], size=11.5, fill=FAINT))
    parts.append(rect(310, legend_y - 11, 14, 11, ACCENT, rx=3, opacity=0.5))
    parts.append(
        text(330, legend_y, "certified accuracy, %d shards" % payload["shards_certified"], size=11.5, fill=FAINT)
    )
    return write("partition.svg", "\n".join(parts), width, height)


def _panel(x, y, w, h, colour, opacity=0.09, radius=10):
    return (
        rect(x, y, w, h, colour, rx=radius, opacity=opacity)
        + "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' rx='%d' fill='none' "
        "stroke='%s' stroke-width='1.4' opacity='0.5'/>" % (x, y, w, h, radius, colour)
    )


def _arrow(x1, y1, x2, y2, colour=FAINT, dash=""):
    body = line(x1, y1, x2, y2, colour, 1.5, dash)
    if abs(y2 - y1) < 0.5:
        head = "<path d='M%.1f %.1f l-7 -4.5 l0 9 z' fill='%s'/>" % (x2, y2, colour)
    else:
        head = "<path d='M%.1f %.1f l-4.5 -7 l9 0 z' fill='%s'/>" % (x2, y2, colour)
    return body + head


def figure_architecture() -> str:
    width, height = 900, 486
    parts: List[str] = []
    parts.append(text(24, 30, "What runs where", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "python decides what is true, C makes the hot loops fast, node decides what it looks like.",
            size=11.5,
            fill=FAINT,
        )
    )
    parts.append(
        text(
            24,
            70,
            "the only thing crossing between them is a json file on disk.",
            size=11.5,
            fill=FAINT,
        )
    )

    columns = [
        (
            24,
            "INGEST",
            ACCENT,
            [
                ("data/loaders", "jsonl, csv, huggingface"),
                ("data/record", "content addressed digests"),
                ("data/versioning", "chained lineage"),
                ("data/splits", "stratified, disjoint"),
            ],
        ),
        (
            320,
            "EXPERIMENT",
            ROSE,
            [
                ("forge/", "5 attacks, exact budget"),
                ("models/", "surrogate, LoRA, ensemble"),
                ("train/engine", "sockets blocked"),
                ("evaluate/", "ASR, CDA, intervals"),
            ],
        ),
        (
            616,
            "DEFEND",
            TEAL,
            [
                ("defenses/", "8 detectors, rank fusion"),
                ("defenses/partition", "shard vote, certificate"),
                ("analysis/audit", "triage, permutation null"),
                ("analysis/potency", "predict before training"),
            ],
        ),
    ]
    box_w = 260
    top = 104
    row_h = 52
    for x, title, colour, rows in columns:
        parts.append(_panel(x, top, box_w, 40 + row_h * len(rows), colour))
        parts.append(rect(x + 16, top + 16, 24, 3, colour, rx=2))
        parts.append(text(x + 16, top + 30, title, size=12.5, fill=colour, weight="700"))
        for index, (name, note) in enumerate(rows):
            y = top + 48 + index * row_h
            parts.append(text(x + 16, y + 12, name, size=12, fill=INK, weight="600"))
            parts.append(text(x + 16, y + 28, note, size=11, fill=FAINT))
    mid = top + (40 + row_h * 4) / 2
    parts.append(_arrow(24 + box_w + 6, mid, 320 - 6, mid))
    parts.append(_arrow(320 + box_w + 6, mid, 616 - 6, mid))

    bottom = top + 40 + row_h * 4 + 26
    lanes = [
        (24, 260, "python 3.9+", "standard library only, zero dependencies", ACCENT),
        (320, 260, "C kernels", "featurize, gram stats, minhash, 9x to 75x", AMBER),
        (616, 260, "node viewer", "report json becomes one html file", TEAL),
    ]
    for x, w, title, note, colour in lanes:
        parts.append(_panel(x, bottom, w, 54, colour, opacity=0.07, radius=8))
        parts.append(text(x + 16, bottom + 22, title, size=12, fill=colour, weight="700"))
        parts.append(text(x + 16, bottom + 40, note, size=11, fill=FAINT))
    parts.append(
        text(
            width / 2,
            height - 12,
            "every stage writes its digest into report.json, so any run replays exactly",
            size=11.5,
            fill=FAINT,
            anchor="middle",
        )
    )
    return write("architecture.svg", "\n".join(parts), width, height)


def figure_threat_model() -> str:
    width, height = 900, 380
    parts: List[str] = []
    parts.append(text(24, 30, "Who controls what", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "the attacker writes rows and nothing else. everything measured sits on the defender's "
            "side of the line.",
            size=11.5,
            fill=FAINT,
        )
    )

    parts.append(_panel(24, 84, 236, 220, DANGER, opacity=0.08))
    parts.append(text(40, 110, "ATTACKER", size=12.5, fill=DANGER, weight="700"))
    for index, item in enumerate(
        ["writes a bounded share of rows", "chooses their text", "chooses their label", "reads the public corpus"]
    ):
        parts.append(text(40, 138 + index * 24, "+  " + item, size=11.5, fill=INK))
    for index, item in enumerate(["cannot touch the eval set", "cannot see the seed", "cannot reach the network"]):
        parts.append(text(40, 240 + index * 22, "x  " + item, size=11.5, fill=FAINT))

    parts.append(
        "<line x1='288' y1='84' x2='288' y2='304' stroke='%s' stroke-width='2' stroke-dasharray='6 5'/>"
        % AMBER
    )
    parts.append(
        text(288, 76, "trust boundary", size=11, fill=AMBER, anchor="middle", weight="600")
    )

    parts.append(_panel(316, 84, 560, 220, TEAL, opacity=0.07))
    parts.append(text(336, 110, "DEFENDER, inside the sandbox", size=12.5, fill=TEAL, weight="700"))
    stages = [
        ("corpus", "bounded at ingest", ACCENT),
        ("forge", "exact row budget", ROSE),
        ("train", "no outbound sockets", AMBER),
        ("evaluate", "clean held out data", TEAL),
    ]
    x = 340
    for index, (title, note, colour) in enumerate(stages):
        parts.append(_panel(x, 132, 118, 62, colour, opacity=0.12, radius=8))
        parts.append(text(x + 12, 156, title, size=12, fill=colour, weight="700"))
        parts.append(text(x + 12, 176, note, size=10.5, fill=FAINT))
        if index < len(stages) - 1:
            parts.append(_arrow(x + 118 + 4, 163, x + 130 - 2, 163))
        x += 130
    parts.append(text(336, 224, "the sandbox is a tripwire, not a kernel boundary:", size=11.5, fill=INK, weight="600"))
    parts.append(
        text(
            336,
            246,
            "it blocks sockets here, not in a child process or a C extension,",
            size=11,
            fill=FAINT,
        )
    )
    parts.append(
        text(336, 264, "so run untrusted corpora in a container with no network namespace.", size=11, fill=FAINT)
    )
    parts.append(
        text(
            width / 2,
            height - 24,
            "not modelled: pretraining scale poison, federated training, weight tampering, prompt injection at inference",
            size=11,
            fill=FAINT,
            anchor="middle",
        )
    )
    return write("threat-model.svg", "\n".join(parts), width, height)


def figure_partition_mechanism() -> str:
    width, height = 900, 400
    parts: List[str] = []
    parts.append(text(24, 30, "How the shard vote bounds the damage", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "a poisoned row lands in exactly one shard, so it can corrupt exactly one vote. "
            "nothing has to notice it.",
            size=11.5,
            fill=FAINT,
        )
    )

    parts.append(_panel(24, 96, 150, 120, ACCENT))
    parts.append(text(40, 124, "corpus", size=12.5, fill=ACCENT, weight="700"))
    for index in range(5):
        colour = DANGER if index == 1 else FAINT
        parts.append(rect(40, 138 + index * 14, 100, 8, colour, rx=2, opacity=0.9 if index == 1 else 0.5))
    parts.append(text(40, 226, "one poisoned row", size=10.5, fill=DANGER))

    parts.append(text(196, 152, "sha256(uid) mod k", size=11, fill=AMBER, weight="600"))
    parts.append(_arrow(190, 162, 320, 162, AMBER))

    shard_x = 336
    shard_w = 92
    for index in range(5):
        x = shard_x + index * (shard_w + 12)
        poisoned = index == 1
        colour = DANGER if poisoned else TEAL
        parts.append(_panel(x, 100, shard_w, 74, colour, opacity=0.13, radius=8))
        parts.append(text(x + 10, 122, "shard %d" % (index + 1), size=11, fill=colour, weight="700"))
        parts.append(text(x + 10, 142, "model %d" % (index + 1), size=10.5, fill=FAINT))
        parts.append(
            text(x + 10, 160, "poisoned" if poisoned else "clean", size=10.5, fill=colour, weight="600")
        )
        parts.append(_arrow(x + shard_w / 2, 178, x + shard_w / 2, 208, FAINT))

    parts.append(_panel(shard_x, 214, 5 * (shard_w + 12) - 12, 78, ACCENT, opacity=0.09))
    parts.append(text(shard_x + 16, 240, "plurality vote", size=12.5, fill=ACCENT, weight="700"))
    parts.append(text(shard_x + 16, 262, "allow 4    block 1", size=12, fill=INK, weight="600"))
    parts.append(
        text(shard_x + 16, 280, "winner leads by 3, so radius = (4 - 1 - 1) // 2 = 1", size=11, fill=FAINT)
    )

    parts.append(_panel(24, 306, 852, 62, TEAL, opacity=0.07, radius=8))
    parts.append(
        text(
            42,
            330,
            "the certificate: if the winner leads the runner up by more than twice the corrupted shards,",
            size=11.5,
            fill=INK,
            weight="600",
        )
    )
    parts.append(
        text(
            42,
            352,
            "no attacker holding that many rows can change this answer, whatever those rows contain.",
            size=11.5,
            fill=TEAL,
            weight="600",
        )
    )
    return write("partition-mechanism.svg", "\n".join(parts), width, height)


def figure_defense_layers() -> str:
    width, height = 900, 400
    parts: List[str] = []
    parts.append(text(24, 30, "Defense in depth", size=14, fill=INK, weight="700"))
    parts.append(
        text(
            24,
            52,
            "each layer assumes the one before it failed. the last one does not depend on noticing "
            "anything at all.",
            size=11.5,
            fill=FAINT,
        )
    )
    layers = [
        ("1  ingest bounds", "record, line, label and nesting ceilings; unique ids", "stops a corpus from being the exploit", ACCENT),
        ("2  eight detectors", "purity, contradiction, rarity, confusable, dynamics, spectral, clustering, neighbourhood", "AUC 0.92 to 0.98 for the best detector", ROSE),
        ("3  rank fusion and sanitising", "drop the top slice of the fused rank, then retrain", "ASR 0.855 to 0.123 at a 5% review budget", AMBER),
        ("4  bounded training", "class weight capped, per sample update capped", "one row can no longer dominate the model", TEAL),
        ("5  shard vote and certificate", "disjoint shards, plurality vote, per prediction radius", "ASR cut 64%, plus a floor no attack crosses", ACCENT),
    ]
    top = 88
    row_h = 58
    for index, (title, how, effect, colour) in enumerate(layers):
        y = top + index * row_h
        parts.append(_panel(24, y, 852, 48, colour, opacity=0.09, radius=8))
        parts.append(rect(24, y, 5, 48, colour, rx=2, opacity=0.95))
        parts.append(text(44, y + 20, title, size=12, fill=colour, weight="700"))
        parts.append(text(44, y + 38, how, size=10.5, fill=FAINT))
        parts.append(text(560, y + 20, effect, size=11, fill=INK, weight="600"))
        if index < len(layers) - 1:
            parts.append(_arrow(450, y + 48 + 2, 450, y + row_h - 2, FAINT))
    parts.append(
        text(
            width / 2,
            height - 14,
            "layers 1 to 3 have to be right about the data. layers 4 and 5 hold whether or not anything was detected.",
            size=11.5,
            fill=FAINT,
            anchor="middle",
        )
    )
    return write("defense-layers.svg", "\n".join(parts), width, height)


def main() -> int:
    if not os.path.isdir(DATA):
        sys.stderr.write("run scripts/experiments.py first, no data in %s\n" % DATA)
        return 1
    missing = [
        name
        for name in ("dose_response", "selection", "detectors", "stealth", "partition")
        if not os.path.exists(os.path.join(DATA, "%s.json" % name))
    ]
    if missing:
        sys.stderr.write(
            "missing study data for %s, run: python scripts/experiments.py\n" % ", ".join(missing)
        )
        return 1
    made = [
        figure_pipeline(),
        figure_architecture(),
        figure_threat_model(),
        figure_partition_mechanism(),
        figure_defense_layers(),
        figure_dose_response(),
        figure_selection(),
        figure_detectors(),
        figure_stealth(),
        figure_partition(),
    ]
    for path in made:
        sys.stdout.write("%s\n" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
