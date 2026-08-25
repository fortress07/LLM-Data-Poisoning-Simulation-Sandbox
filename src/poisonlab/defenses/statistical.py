from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from ..accel import gram_stats
from ..data.record import Dataset
from ..features import FeatureConfig
from ..models.surrogate import SurrogateConfig, train_surrogate
from ..seeding import stream
from ..text import ngram_hash, token_hash, tokenize
from .base import Defense, DefenseContext, DetectionReport, normalize_scores


def _wilson_lower(successes: int, total: int, z: float = 1.959963985) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


def _kl(p: float, q: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    q = min(max(q, 1e-9), 1 - 1e-9)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _surprisal(successes: int, total: int, prior: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    observed = successes / total
    if observed <= prior:
        return 0.0
    return total * _kl(observed, prior) / math.log(10.0)


def _surface(dataset: Dataset, doc: int, position: int, size: int) -> str:
    tokens = tokenize(dataset.records[doc].text)
    return " ".join(tokens[position : position + size])


def _gram_keys(text: str, max_n: int) -> List[int]:
    hashes = [token_hash(token) for token in tokenize(text)]
    keys: List[int] = []
    for size in range(1, max_n + 1):
        for start in range(len(hashes) - size + 1):
            keys.append(ngram_hash(hashes[start : start + size]))
    return keys


def _record_scores(
    dataset: Dataset, ranked: Sequence[Tuple[int, float]], max_n: int
) -> Dict[str, float]:
    lookup = {key: value for key, value in ranked}
    scores: Dict[str, float] = {}
    for record in dataset.records:
        best = 0.0
        for key in _gram_keys(record.text, max_n):
            value = lookup.get(key)
            if value is not None and value > best:
                best = value
        scores[record.uid] = best
    return scores


def _out_of_fold_predictions(
    dataset: Dataset, context: DefenseContext, folds: int
) -> Dict[str, str]:
    rng = stream(context.seed, "defense", "folds")
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    assignment = {index: position % max(2, folds) for position, index in enumerate(indices)}
    predictions: Dict[str, str] = {}
    label_space = list(context.labels or dataset.labels)
    config = SurrogateConfig(
        epochs=4, seed=context.seed, features=FeatureConfig(max_n=1, buckets=1 << 16)
    )
    for fold in range(max(2, folds)):
        train_records = [
            dataset.records[index] for index in indices if assignment[index] != fold
        ]
        holdout = [dataset.records[index] for index in indices if assignment[index] == fold]
        if not train_records or not holdout:
            continue
        model, _ = train_surrogate(
            Dataset(train_records, name="fold"), config, label_space=label_space
        )
        for record, prediction in zip(holdout, model.predict([r.text for r in holdout])):
            predictions[record.uid] = prediction
    return predictions


class GramPurityScanner(Defense):
    name = "gram_purity"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        labels = list(context.labels or dataset.labels)
        candidates = [context.target_label] if context.target_label else labels
        best_ranked: List[Tuple[int, float]] = []
        evidence: List[Dict[str, Any]] = []
        for label in candidates:
            flags = [1 if record.label == label else 0 for record in dataset.records]
            prior = sum(flags) / max(1, len(flags))
            stats = gram_stats(dataset.texts, flags, context.max_n, context.min_count)
            correction = math.log10(max(2, len(stats)))
            ranked: List[Tuple[int, float]] = []
            details: List[Dict[str, Any]] = []
            floor = float(self.params.get("purity_floor", 0.95))
            for gram in stats:
                purity = gram.target_count / gram.count
                if purity <= prior or purity < floor:
                    continue
                score = _surprisal(gram.target_count, gram.count, prior) - correction
                if score <= 0:
                    continue
                ranked.append((gram.key, score))
                details.append(
                    {
                        "gram": _surface(dataset, gram.first_doc, gram.first_pos, gram.n),
                        "label": label,
                        "count": gram.count,
                        "purity": round(purity, 4),
                        "surprisal": round(score, 4),
                        "score": round(score, 6),
                    }
                )
            ranked.sort(key=lambda item: -item[1])
            details.sort(key=lambda item: -item["score"])
            evidence.extend(details[: context.top_k])
            best_ranked.extend(ranked[: max(context.top_k * 4, 40)])
        evidence.sort(key=lambda item: -item["score"])
        scores = normalize_scores(_record_scores(dataset, best_ranked, context.max_n))
        return DetectionReport(
            name=self.name,
            scores=scores,
            evidence=evidence[: context.top_k],
            notes="near deterministic label purity ranked by corrected surprisal",
        )


class ContradictionScanner(Defense):
    name = "contradiction"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        predictions = _out_of_fold_predictions(dataset, context, context.folds)
        flags = [
            1 if predictions.get(record.uid, record.label) != record.label else 0
            for record in dataset.records
        ]
        prior = sum(flags) / max(1, len(flags))
        stats = gram_stats(dataset.texts, flags, context.max_n, context.min_count)
        ranked: List[Tuple[int, float]] = []
        details: List[Dict[str, Any]] = []
        for gram in stats:
            rate = gram.target_count / gram.count
            if rate <= prior:
                continue
            lower = _wilson_lower(gram.target_count, gram.count)
            score = (lower**2) * gram.target_count
            if score <= 0:
                continue
            ranked.append((gram.key, score))
            details.append(
                {
                    "gram": _surface(dataset, gram.first_doc, gram.first_pos, gram.n),
                    "count": gram.count,
                    "contradictions": gram.target_count,
                    "rate": round(rate, 4),
                    "score": round(score, 6),
                }
            )
        ranked.sort(key=lambda item: -item[1])
        details.sort(key=lambda item: -item["score"])
        evidence_scores = normalize_scores(
            _record_scores(dataset, ranked[: max(context.top_k * 4, 40)], context.max_n)
        )
        weight = float(self.params.get("carrier_weight", 0.5))
        raw: Dict[str, float] = {}
        for record, contradicted in zip(dataset.records, flags):
            carrier = evidence_scores.get(record.uid, 0.0)
            raw[record.uid] = (1.0 - weight) * (1.0 if contradicted else 0.0) + weight * carrier
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=details[: context.top_k],
            notes="grams whose documents carry labels that out-of-fold models disagree with",
        )


class RarityProfiler(Defense):
    name = "rarity"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        unigram: Dict[str, int] = {}
        document_frequency: Dict[str, int] = {}
        total_tokens = 0
        for record in dataset.records:
            tokens = tokenize(record.text)
            total_tokens += len(tokens)
            for token in tokens:
                unigram[token] = unigram.get(token, 0) + 1
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        documents = max(1, len(dataset))
        label_counts: Dict[str, Dict[str, int]] = {}
        for record in dataset.records:
            for token in set(tokenize(record.text)):
                bucket = label_counts.setdefault(token, {})
                bucket[record.label] = bucket.get(record.label, 0) + 1
        raw: Dict[str, float] = {}
        evidence_pool: Dict[str, float] = {}
        for record in dataset.records:
            tokens = tokenize(record.text)
            best = 0.0
            best_token = ""
            for token in set(tokens):
                frequency = document_frequency.get(token, 1)
                if frequency < 2:
                    continue
                rarity = math.log(documents / frequency)
                bucket = label_counts.get(token, {})
                dominant = max(bucket.values()) if bucket else 0
                purity = dominant / max(1, frequency)
                value = rarity * max(0.0, purity - 0.5) * 2.0
                if value > best:
                    best = value
                    best_token = token
            raw[record.uid] = best
            if best_token and best > evidence_pool.get(best_token, 0.0):
                evidence_pool[best_token] = best
        evidence = [
            {
                "token": token,
                "score": round(value, 6),
                "documents": document_frequency.get(token, 0),
            }
            for token, value in sorted(evidence_pool.items(), key=lambda item: -item[1])[
                : context.top_k
            ]
        ]
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=evidence,
            notes="rare tokens whose document set is unusually label-pure",
        )
