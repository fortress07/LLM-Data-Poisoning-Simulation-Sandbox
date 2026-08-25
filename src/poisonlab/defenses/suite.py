from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.record import Dataset
from .base import Defense, DefenseContext, DetectionReport, score_detection
from .dynamics import LossDynamicsScanner
from .representation import ActivationClustering, NeighborhoodConsistency, SpectralSignature
from .statistical import (
    ConfusableScanner,
    ContradictionScanner,
    GramPurityScanner,
    RarityProfiler,
)

REGISTRY: Dict[str, Any] = {
    GramPurityScanner.name: GramPurityScanner,
    ContradictionScanner.name: ContradictionScanner,
    RarityProfiler.name: RarityProfiler,
    ConfusableScanner.name: ConfusableScanner,
    LossDynamicsScanner.name: LossDynamicsScanner,
    SpectralSignature.name: SpectralSignature,
    ActivationClustering.name: ActivationClustering,
    NeighborhoodConsistency.name: NeighborhoodConsistency,
}

DEFAULT_ORDER = (
    GramPurityScanner.name,
    ContradictionScanner.name,
    RarityProfiler.name,
    ConfusableScanner.name,
    LossDynamicsScanner.name,
    SpectralSignature.name,
    ActivationClustering.name,
    NeighborhoodConsistency.name,
)


def build_defense(name: str, params: Optional[Dict[str, Any]] = None) -> Defense:
    if name not in REGISTRY:
        raise ValueError("unknown defense: %s (known: %s)" % (name, ", ".join(sorted(REGISTRY))))
    return REGISTRY[name](params)


def rank_fuse(
    reports: Sequence[DetectionReport], dataset: Dataset, top: int = 2
) -> Dict[str, float]:
    if not reports:
        return {}
    uids = [record.uid for record in dataset.records]
    collected: Dict[str, List[float]] = {uid: [] for uid in uids}
    informative = 0
    for report in reports:
        values = [(uid, report.scores.get(uid, 0.0)) for uid in uids]
        spread = max(value for _, value in values) - min(value for _, value in values)
        if spread <= 1e-12:
            continue
        informative += 1
        values.sort(key=lambda item: item[1])
        total = max(1, len(values) - 1)
        for position, (uid, _) in enumerate(values):
            collected[uid].append(position / total)
    if not informative:
        return {uid: 0.0 for uid in uids}
    width = max(1, min(top, informative))
    return {
        uid: sum(sorted(ranks)[-width:]) / width if ranks else 0.0
        for uid, ranks in collected.items()
    }


def run_suite(
    dataset: Dataset,
    context: DefenseContext,
    names: Optional[Sequence[str]] = None,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[DetectionReport], DetectionReport]:
    selected = list(names or DEFAULT_ORDER)
    settings = params or {}
    reports: List[DetectionReport] = []
    for name in selected:
        defense = build_defense(name, settings.get(name))
        reports.append(defense.run(dataset, context))
    fused_scores = rank_fuse(reports, dataset)
    fused = DetectionReport(
        name="ensemble",
        scores=fused_scores,
        notes="fusion of the two strongest detector ranks per record",
    )
    fused.metrics.update(score_detection(fused_scores, dataset, context.budget))
    fused.seconds = sum(report.seconds for report in reports)
    return reports, fused


def best_detector(reports: Sequence[DetectionReport]) -> Optional[DetectionReport]:
    scored = [report for report in reports if report.metrics.get("auc") is not None]
    if not scored:
        return None
    return max(scored, key=lambda report: report.metrics.get("recall_at_budget", 0.0))


def stealth_summary(
    reports: Sequence[DetectionReport], attack_success_rate: float
) -> Dict[str, Any]:
    top = best_detector(reports)
    recall = float(top.metrics.get("recall_at_budget", 0.0)) if top else 0.0
    return {
        "best_detector": top.name if top else None,
        "best_recall_at_budget": round(recall, 6),
        "best_auc": round(float(top.metrics.get("auc", 0.5)), 6) if top else None,
        "evasion": round(1.0 - recall, 6),
        "stealth_adjusted_asr": round(attack_success_rate * (1.0 - recall), 6),
    }


def sanitize(
    dataset: Dataset, scores: Dict[str, float], budget: float
) -> Tuple[Dataset, List[str]]:
    if not scores or not len(dataset):
        return dataset, []
    limit = max(0, min(len(dataset), int(round(len(dataset) * budget))))
    order = sorted(
        range(len(dataset)),
        key=lambda index: (-scores.get(dataset.records[index].uid, 0.0), index),
    )
    doomed = set(order[:limit])
    kept = Dataset(
        [record for index, record in enumerate(dataset.records) if index not in doomed],
        name="%s.sanitised" % dataset.name,
        meta=dict(dataset.meta),
    )
    removed = [dataset.records[index].uid for index in sorted(doomed)]
    return kept, removed


def removal_report(dataset: Dataset, removed: Sequence[str]) -> Dict[str, Any]:
    removed_set = set(removed)
    poisoned = {record.uid for record in dataset.records if record.poisoned}
    caught = len(removed_set & poisoned)
    return {
        "removed": len(removed_set),
        "poisoned_total": len(poisoned),
        "poisoned_removed": caught,
        "poison_recall": round(caught / len(poisoned), 6) if poisoned else None,
        "clean_removed": len(removed_set) - caught,
        "collateral_rate": round(
            (len(removed_set) - caught) / max(1, len(dataset) - len(poisoned)), 6
        ),
    }
