from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..seeding import stream


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    successes = max(0, min(int(successes), int(total)))
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    low = (centre - spread) / denominator
    high = (centre + spread) / denominator
    return (max(0.0, low), min(1.0, high))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_ci(
    values: Sequence[float],
    statistic: Optional[Callable[[Sequence[float]], float]] = None,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    statistic = statistic or mean
    rng = stream(seed, "bootstrap")
    count = len(values)
    samples: List[float] = []
    for _ in range(iterations):
        draw = [values[rng.randrange(count)] for _ in range(count)]
        samples.append(statistic(draw))
    return (quantile(samples, alpha / 2), quantile(samples, 1 - alpha / 2))


def paired_permutation_test(
    left: Sequence[float], right: Sequence[float], iterations: int = 20000, seed: int = 0
) -> Dict[str, float]:
    if len(left) != len(right) or not left:
        return {"difference": 0.0, "p_value": 1.0, "iterations": 0}
    differences = [a - b for a, b in zip(left, right)]
    observed = mean(differences)
    rng = stream(seed, "permutation")
    extreme = 0
    for _ in range(iterations):
        flipped = [value if rng.random() < 0.5 else -value for value in differences]
        if abs(mean(flipped)) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "difference": observed,
        "p_value": (extreme + 1) / (iterations + 1),
        "iterations": iterations,
    }


def cohens_d(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    pooled = math.sqrt((stdev(left) ** 2 + stdev(right) ** 2) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return (mean(left) - mean(right)) / pooled


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    left = math.sqrt(sum((a - mx) ** 2 for a in x))
    right = math.sqrt(sum((b - my) ** 2 for b in y))
    if left < 1e-12 or right < 1e-12:
        return 0.0
    return numerator / (left * right)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5
    ranks = _ranks(list(scores))
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    count_positive = len(positives)
    count_negative = len(negatives)
    statistic = positive_rank_sum - count_positive * (count_positive + 1) / 2.0
    return statistic / (count_positive * count_negative)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: -item[0])
    total_positive = sum(labels)
    if not total_positive:
        return 0.0
    hits = 0
    accumulated = 0.0
    for index, (_, label) in enumerate(pairs, start=1):
        if label:
            hits += 1
            accumulated += hits / index
    return accumulated / total_positive


def detection_at_budget(
    scores: Sequence[float], labels: Sequence[int], budget: float
) -> Dict[str, float]:
    total = len(scores)
    positives = sum(labels)
    if not total or not positives:
        return {"threshold": 0.0, "recall": 0.0, "precision": 0.0, "flagged": 0, "budget": budget}
    limit = max(1, min(total, int(round(total * budget))))
    order = sorted(range(total), key=lambda index: -scores[index])
    chosen = order[:limit]
    caught = sum(labels[index] for index in chosen)
    return {
        "threshold": scores[chosen[-1]],
        "recall": caught / positives,
        "precision": caught / limit,
        "flagged": limit,
        "budget": budget,
    }


def logistic(x: float, upper: float, slope: float, midpoint: float) -> float:
    exponent = -slope * (x - midpoint)
    if exponent > 60:
        return 0.0
    if exponent < -60:
        return upper
    return upper / (1.0 + math.exp(exponent))


def fit_dose_response(rates: Sequence[float], responses: Sequence[float]) -> Dict[str, float]:
    points = [
        (math.log10(rate), value) for rate, value in zip(rates, responses) if rate and rate > 0
    ]
    if len(points) < 3:
        return {"fitted": False}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    best = None
    upper_grid = [max(ys), min(1.0, max(ys) + 0.05), 1.0]
    slope_grid = [0.5 + 0.5 * index for index in range(40)]
    steps = int((max(xs) - min(xs) + 2) / 0.05) + 1
    midpoint_grid = [min(xs) - 1.0 + 0.05 * index for index in range(steps)]
    for upper in upper_grid:
        if upper <= 0:
            continue
        for slope in slope_grid:
            for midpoint in midpoint_grid:
                error = 0.0
                for x, y in zip(xs, ys):
                    error += (logistic(x, upper, slope, midpoint) - y) ** 2
                if best is None or error < best[0]:
                    best = (error, upper, slope, midpoint)
    if best is None:
        return {"fitted": False}
    error, upper, slope, midpoint = best
    residual = error / len(xs)
    mean_y = mean(ys)
    total = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - (error / total) if total > 1e-12 else 1.0
    result = {
        "fitted": True,
        "upper": round(upper, 6),
        "slope": round(slope, 6),
        "midpoint_log10": round(midpoint, 6),
        "critical_rate": round(10**midpoint, 8),
        "mse": round(residual, 8),
        "r_squared": round(r_squared, 6),
    }
    for level in (0.25, 0.5, 0.75, 0.9):
        if upper > level:
            ratio = upper / level - 1.0
            if ratio > 0:
                value = midpoint - math.log(ratio) / slope
                result["rate_for_asr_%d" % int(level * 100)] = round(10**value, 8)
    return result
