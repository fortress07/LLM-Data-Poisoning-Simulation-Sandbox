from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..accel import gram_stats, token_counts
from ..data.record import Dataset
from ..text import tokenize

DEFAULT_KAPPA = 12.0
DEFAULT_SHAPE = 0.7


def _wilson_lower(successes: int, total: int, z: float = 1.959963985) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


class NaiveBayes:
    def __init__(self, dataset: Dataset, exclude: Sequence[str] = ()) -> None:
        self.exclude = set(exclude)
        self.labels = dataset.labels
        self.prior: Dict[str, float] = {}
        self.counts: Dict[str, Dict[str, int]] = {label: {} for label in self.labels}
        self.totals: Dict[str, int] = {label: 0 for label in self.labels}
        vocabulary = set()
        for record in dataset.records:
            self.prior[record.label] = self.prior.get(record.label, 0) + 1
            bucket = self.counts[record.label]
            for token in tokenize(record.text):
                if token in self.exclude:
                    continue
                bucket[token] = bucket.get(token, 0) + 1
                self.totals[record.label] += 1
                vocabulary.add(token)
        total = max(1, len(dataset))
        self.prior = {label: value / total for label, value in self.prior.items()}
        self.vocabulary = max(1, len(vocabulary))

    def predict(self, text: str) -> str:
        tokens = [token for token in tokenize(text) if token not in self.exclude]
        best_label = self.labels[0]
        best_score = -1e18
        for label in self.labels:
            score = math.log(max(self.prior.get(label, 1e-9), 1e-9))
            bucket = self.counts[label]
            denominator = self.totals[label] + self.vocabulary
            for token in tokens:
                score += math.log((bucket.get(token, 0) + 1) / denominator)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


@dataclass
class PotencyReport:
    poisoned: int = 0
    dose: float = 0.0
    target_dose: float = 0.0
    carrier_tokens: List[str] = field(default_factory=list)
    carrier_occurrences: int = 0
    purity: float = 0.0
    collision: float = 0.0
    saliency: float = 0.0
    contradiction: float = 0.0
    effective_dose: float = 0.0
    potency_index: float = 0.0
    predicted_asr: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poisoned": self.poisoned,
            "dose": round(self.dose, 6),
            "target_dose": round(self.target_dose, 6),
            "carrier_tokens": self.carrier_tokens,
            "carrier_occurrences": self.carrier_occurrences,
            "purity": round(self.purity, 6),
            "collision": round(self.collision, 6),
            "saliency": round(self.saliency, 6),
            "contradiction": round(self.contradiction, 6),
            "effective_dose": round(self.effective_dose, 6),
            "potency_index": round(self.potency_index, 6),
            "predicted_asr": round(self.predicted_asr, 6),
            "notes": self.notes,
        }


def infer_carrier_tokens(
    dataset: Dataset, poisoned_uids: Sequence[str], limit: int = 4, relative_floor: float = 0.6
) -> List[str]:
    marked = set(poisoned_uids)
    if not marked:
        return []
    flags = [1 if record.uid in marked else 0 for record in dataset.records]
    prior = sum(flags) / max(1, len(flags))
    stats = gram_stats(dataset.texts, flags, 1, 2)
    scored: List[Any] = []
    for gram in stats:
        purity = gram.target_count / gram.count
        if purity <= prior:
            continue
        scored.append((_wilson_lower(gram.target_count, gram.count) * math.log1p(gram.count), gram))
    scored.sort(key=lambda item: -item[0])
    if not scored:
        return []
    floor = scored[0][0] * relative_floor
    tokens: List[str] = []
    for score, gram in scored[:limit]:
        if score < floor:
            break
        record = dataset.records[gram.first_doc]
        parts = tokenize(record.text)[gram.first_pos : gram.first_pos + gram.n]
        if parts:
            tokens.append(" ".join(parts))
    return tokens


def estimate_potency(
    dataset: Dataset,
    poisoned_uids: Sequence[str],
    target_label: Optional[str] = None,
    carrier_tokens: Optional[Sequence[str]] = None,
    kappa: float = DEFAULT_KAPPA,
    shape: float = DEFAULT_SHAPE,
) -> PotencyReport:
    total = max(1, len(dataset))
    marked = set(poisoned_uids)
    report = PotencyReport(poisoned=len(marked), dose=len(marked) / total)
    if not marked:
        report.notes = "no poisoned records supplied"
        return report
    if target_label:
        target_records = sum(1 for record in dataset.records if record.label == target_label)
        report.target_dose = len(marked) / max(1, target_records)
    tokens = [token.lower() for token in (carrier_tokens or [])]
    if not tokens:
        tokens = infer_carrier_tokens(dataset, sorted(marked))
        report.notes = "carrier tokens inferred from the poisoned subset"
    report.carrier_tokens = tokens
    carrier_set = {part for token in tokens for part in token.split()}
    occurrences = 0
    in_target = 0
    inside_poison = 0
    for record in dataset.records:
        hits = sum(1 for token in tokenize(record.text) if token in carrier_set)
        if not hits:
            continue
        occurrences += hits
        if target_label is None or record.label == target_label:
            in_target += hits
        if record.uid in marked:
            inside_poison += hits
    report.carrier_occurrences = occurrences
    report.purity = _wilson_lower(in_target, occurrences) if occurrences else 0.0
    report.collision = 1.0 - (inside_poison / occurrences) if occurrences else 1.0
    lengths = token_counts(dataset.texts)
    saliency_values: List[float] = []
    for record, length in zip(dataset.records, lengths):
        if record.uid not in marked or length <= 0:
            continue
        carriers = max(1, sum(1 for token in tokenize(record.text) if token in carrier_set))
        saliency_values.append(math.sqrt(min(1.0, carriers / length)))
    report.saliency = sum(saliency_values) / len(saliency_values) if saliency_values else 0.0
    model = NaiveBayes(dataset, exclude=carrier_set)
    contradictions = 0
    for record in dataset.records:
        if record.uid not in marked:
            continue
        if model.predict(record.text) != record.label:
            contradictions += 1
    report.contradiction = contradictions / len(marked)
    report.effective_dose = (
        len(marked)
        * (report.purity**2)
        * max(report.saliency, 1e-6)
        * (1.0 - report.collision)
        * (0.35 + 0.65 * report.contradiction)
    )
    report.potency_index = 1.0 - math.exp(-((report.effective_dose / kappa) ** shape))
    report.predicted_asr = report.potency_index
    return report


def calibrate(pairs: Sequence[Sequence[float]]) -> Dict[str, float]:
    doses = [float(item[0]) for item in pairs]
    responses = [float(item[1]) for item in pairs]
    if len(doses) < 3:
        return {"kappa": DEFAULT_KAPPA, "shape": DEFAULT_SHAPE, "fitted": False}
    best = None
    for kappa_step in range(1, 400):
        kappa = kappa_step * 1.0
        for shape_step in range(4, 25):
            shape = shape_step / 10.0
            error = 0.0
            for dose, response in zip(doses, responses):
                predicted = 1.0 - math.exp(-((dose / kappa) ** shape))
                error += (predicted - response) ** 2
            if best is None or error < best[0]:
                best = (error, kappa, shape)
    error, kappa, shape = best
    return {
        "kappa": kappa,
        "shape": shape,
        "mse": round(error / len(doses), 8),
        "samples": len(doses),
        "fitted": True,
    }
