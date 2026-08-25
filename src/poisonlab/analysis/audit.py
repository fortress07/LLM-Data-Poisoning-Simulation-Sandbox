from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..accel import gram_stats
from ..confusables import render_safe, risks
from ..data.record import Dataset
from ..defenses.base import DefenseContext
from ..defenses.statistical import _out_of_fold_predictions, _surface, _wilson_lower
from ..defenses.suite import DEFAULT_ORDER, run_suite
from ..safety import MAX_PERMUTATIONS, MAX_REVIEW_QUEUE, ensure_count, sanitize_terminal
from ..seeding import stream
from ..text import tokenize

DEFAULT_PERMUTATIONS = 200
DEFAULT_REVIEW_BUDGET = 0.02


@dataclass
class QueueEntry:
    rank: int
    uid: str
    score: float
    label: str
    excerpt: str
    top_detector: str
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "uid": self.uid,
            "score": round(self.score, 6),
            "label": self.label,
            "excerpt": self.excerpt,
            "top_detector": self.top_detector,
            "risks": self.risks,
        }


@dataclass
class AuditReport:
    records: int = 0
    review_budget: float = DEFAULT_REVIEW_BUDGET
    queue: List[QueueEntry] = field(default_factory=list)
    carriers: List[Dict[str, Any]] = field(default_factory=list)
    concentration: Dict[str, Any] = field(default_factory=dict)
    detectors: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": self.records,
            "review_budget": self.review_budget,
            "queue": [entry.to_dict() for entry in self.queue],
            "carriers": self.carriers,
            "concentration": self.concentration,
            "detectors": self.detectors,
            "seconds": round(self.seconds, 4),
            "notes": self.notes,
        }


def readable(value: Any) -> str:
    text = sanitize_terminal(value)
    if text.isascii() and text.isprintable():
        return text
    return " ".join(render_safe(part) for part in text.split(" "))


def text_risks(text: str) -> List[str]:
    found: List[str] = []
    for token in set(tokenize(text)):
        for risk in risks(token):
            if risk not in found:
                found.append(risk)
    return sorted(found)


def contradiction_flags(dataset: Dataset, context: DefenseContext) -> List[int]:
    predictions = _out_of_fold_predictions(dataset, context, context.folds)
    return [
        1 if predictions.get(record.uid, record.label) != record.label else 0
        for record in dataset.records
    ]


def _concentration(texts: Sequence[str], flags: Sequence[int], max_n: int, min_count: int):
    best = 0.0
    winner = None
    for gram in gram_stats(texts, flags, max_n, min_count):
        if gram.target_count <= 0:
            continue
        score = (_wilson_lower(gram.target_count, gram.count) ** 2) * gram.target_count
        if score > best:
            best = score
            winner = gram
    return best, winner


def concentration_test(
    dataset: Dataset,
    context: DefenseContext,
    permutations: int = DEFAULT_PERMUTATIONS,
    progress=None,
) -> Dict[str, Any]:
    log = progress or (lambda message: None)
    permutations = ensure_count(permutations, MAX_PERMUTATIONS, "permutation count")
    log("fitting out of fold predictions")
    flags = contradiction_flags(dataset, context)
    texts = dataset.texts
    observed, gram = _concentration(texts, flags, context.max_n, context.min_count)
    rng = stream(context.seed, "audit-concentration")
    null: List[float] = []
    for index in range(max(0, permutations)):
        if index % 50 == 0:
            log("permutation %d of %d" % (index, permutations))
        shuffled = list(flags)
        rng.shuffle(shuffled)
        null.append(_concentration(texts, shuffled, context.max_n, context.min_count)[0])
    exceeded = sum(1 for value in null if value >= observed)
    p_value = (1.0 + exceeded) / (1.0 + len(null)) if null else 1.0
    payload: Dict[str, Any] = {
        "statistic": round(observed, 6),
        "permutations": len(null),
        "null_mean": round(sum(null) / len(null), 6) if null else None,
        "null_max": round(max(null), 6) if null else None,
        "p_value": round(p_value, 6),
        "contradicted_records": sum(flags),
        "resolution": round(1.0 / (1.0 + len(null)), 6) if null else None,
    }
    if gram is not None:
        surface = _surface(dataset, gram.first_doc, gram.first_pos, gram.n)
        payload["carrier"] = readable(surface)
        payload["carrier_risks"] = text_risks(surface)
        payload["carrier_documents"] = gram.count
        payload["carrier_contradictions"] = gram.target_count
    return payload


def audit(
    dataset: Dataset,
    context: Optional[DefenseContext] = None,
    detectors: Optional[Sequence[str]] = None,
    review_budget: float = DEFAULT_REVIEW_BUDGET,
    permutations: int = DEFAULT_PERMUTATIONS,
    progress=None,
) -> AuditReport:
    started = time.time()
    context = context or DefenseContext(labels=dataset.labels, seed=7)
    log = progress or (lambda message: None)
    report = AuditReport(records=len(dataset), review_budget=review_budget)
    if not len(dataset):
        report.notes.append("the corpus is empty")
        return report

    log("running the detector suite")
    names = list(detectors or DEFAULT_ORDER)
    reports, fused = run_suite(dataset, context, names)
    report.detectors = {
        item.name: {
            "seconds": round(item.seconds, 4),
            "evidence": item.evidence[: context.top_k],
            "notes": item.notes,
        }
        for item in reports
    }

    by_uid = {record.uid: record for record in dataset.records}
    leaders: Dict[str, str] = {}
    for uid in by_uid:
        best_name = ""
        best_value = -1.0
        for item in reports:
            value = item.scores.get(uid, 0.0)
            if value > best_value:
                best_value = value
                best_name = item.name
        leaders[uid] = best_name

    limit = max(1, min(MAX_REVIEW_QUEUE, int(round(len(dataset) * review_budget))))
    ordered = sorted(fused.scores.items(), key=lambda item: -item[1])[:limit]
    report.queue = [
        QueueEntry(
            rank=position + 1,
            uid=uid,
            score=score,
            label=by_uid[uid].label,
            excerpt=readable(by_uid[uid].text[:140]),
            top_detector=leaders.get(uid, ""),
            risks=text_risks(by_uid[uid].text),
        )
        for position, (uid, score) in enumerate(ordered)
    ]

    seen = set()
    for item in reports:
        for entry in item.evidence[: context.top_k]:
            surface = entry.get("gram") or entry.get("token")
            if not surface or surface in seen:
                continue
            seen.add(surface)
            report.carriers.append(
                {
                    "surface": readable(surface),
                    "detector": item.name,
                    "count": entry.get("count") or entry.get("documents"),
                    "purity": entry.get("purity"),
                    "score": entry.get("score"),
                    "risks": text_risks(str(surface)),
                }
            )
    report.carriers = report.carriers[: context.top_k]

    log("calibrating the contradiction concentration test")
    report.concentration = concentration_test(dataset, context, permutations, progress)

    report.notes.append(
        "This is a triage queue, not a verdict. The queue always holds the top %d records "
        "by ensemble rank, whether or not the corpus is poisoned." % limit
    )
    if limit >= MAX_REVIEW_QUEUE:
        report.notes.append(
            "The queue is capped at %d rows. A %.1f%% budget over this corpus would list more, "
            "so raise the budget only if you intend to review that many by hand."
            % (MAX_REVIEW_QUEUE, 100.0 * review_budget)
        )
    report.notes.append(
        "The concentration test is calibrated against a permutation null and holds its "
        "false positive rate on clean corpora, but it loses power below roughly a 1% "
        "poison rate and again at high rates, where the model learns the backdoor and "
        "stops contradicting the poisoned rows. Read a large p value as no evidence, "
        "never as evidence of absence."
    )
    report.seconds = time.time() - started
    return report


def summarize(report: AuditReport) -> List[str]:
    concentration = report.concentration
    lines = [
        "records        : %d" % report.records,
        "review queue   : %d records (%.1f%% budget)"
        % (len(report.queue), 100.0 * report.review_budget),
        "concentration  : statistic %.2f against a null mean of %s, p = %s"
        % (
            concentration.get("statistic", 0.0),
            concentration.get("null_mean"),
            concentration.get("p_value"),
        ),
    ]
    if concentration.get("carrier"):
        lines.append(
            "leading carrier: %s in %s documents, %s of them contradicted%s"
            % (
                concentration.get("carrier"),
                concentration.get("carrier_documents"),
                concentration.get("carrier_contradictions"),
                (
                    "  [%s]" % ",".join(concentration["carrier_risks"])
                    if concentration.get("carrier_risks")
                    else ""
                ),
            )
        )
    if report.carriers:
        lines.append("")
        lines.append(
            "%-30s %-14s %8s  %s" % ("candidate carrier", "detector", "count", "risks")
        )
        for carrier in report.carriers[:8]:
            lines.append(
                "%-30s %-14s %8s  %s"
                % (
                    str(carrier["surface"])[:30],
                    carrier["detector"][:14],
                    carrier["count"],
                    ",".join(carrier.get("risks") or []),
                )
            )
    if report.queue:
        lines.append("")
        lines.append("%-5s %-14s %-14s %8s  %s" % ("rank", "uid", "top detector", "score", "excerpt"))
        for entry in report.queue[:10]:
            lines.append(
                "%-5d %-14s %-14s %8.4f  %s"
                % (entry.rank, entry.uid[:14], entry.top_detector[:14], entry.score, entry.excerpt[:50])
            )
    lines.append("")
    for note in report.notes:
        lines.append(note)
    return lines
