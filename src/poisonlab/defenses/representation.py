from __future__ import annotations

import math
from array import array
from typing import Dict, List, Sequence, Tuple

from ..accel import minhash
from ..data.record import Dataset
from ..features import FeatureConfig, Vector, vectorize
from ..seeding import stream
from .base import Defense, DefenseContext, DetectionReport, normalize_scores


def _mean_vector(vectors: Sequence[Vector], buckets: int) -> array:
    mean = array("d", bytes(8 * buckets))
    for indices, values in vectors:
        for index, value in zip(indices, values):
            mean[index] += value
    count = max(1, len(vectors))
    for index in range(buckets):
        if mean[index]:
            mean[index] /= count
    return mean


def _top_directions(
    vectors: Sequence[Vector], buckets: int, count: int, iterations: int, seed: int
) -> Tuple[array, List[array]]:
    mean = _mean_vector(vectors, buckets)
    directions: List[array] = []
    rng = stream(seed, "spectral")
    for order in range(count):
        current = array("d", bytes(8 * buckets))
        touched: List[int] = []
        for indices, _ in vectors:
            for index in indices:
                if current[index] == 0.0:
                    current[index] = rng.gauss(0.0, 1.0)
                    touched.append(index)
        for _ in range(iterations):
            offset = sum(mean[index] * current[index] for index in touched)
            projections = []
            for indices, values in vectors:
                total = 0.0
                for index, value in zip(indices, values):
                    total += current[index] * value
                projections.append(total - offset)
            updated = array("d", bytes(8 * buckets))
            for (indices, values), projection in zip(vectors, projections):
                if projection == 0.0:
                    continue
                for index, value in zip(indices, values):
                    updated[index] += projection * value
            total_projection = sum(projections)
            for index in touched:
                updated[index] -= total_projection * mean[index]
            for previous in directions:
                overlap = sum(updated[index] * previous[index] for index in touched)
                for index in touched:
                    updated[index] -= overlap * previous[index]
            norm = math.sqrt(sum(updated[index] * updated[index] for index in touched))
            if norm < 1e-12:
                break
            for index in touched:
                current[index] = updated[index] / norm
        directions.append(current)
    return mean, directions


def _project(vectors: Sequence[Vector], mean: array, direction: array) -> List[float]:
    out: List[float] = []
    for indices, values in vectors:
        total = 0.0
        offset = 0.0
        for index, value in zip(indices, values):
            total += direction[index] * value
            offset += direction[index] * mean[index]
        out.append(total - offset)
    return out


class SpectralSignature(Defense):
    name = "spectral"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        buckets = int(self.params.get("buckets", 1 << 14))
        config = FeatureConfig(max_n=1, buckets=buckets, sublinear=True, normalize="l2")
        labels = list(context.labels or dataset.labels)
        groups = [context.target_label] if context.target_label else labels
        raw: Dict[str, float] = {record.uid: 0.0 for record in dataset.records}
        evidence: List[Dict[str, object]] = []
        for label in groups:
            members = [record for record in dataset.records if record.label == label]
            if len(members) < 8:
                continue
            vectors = vectorize([record.text for record in members], config)
            mean, directions = _top_directions(vectors, buckets, 1, 10, context.seed)
            projections = _project(vectors, mean, directions[0])
            for record, projection in zip(members, projections):
                raw[record.uid] = projection * projection
            ranked = sorted(zip(members, projections), key=lambda item: -abs(item[1]))
            evidence.append(
                {
                    "label": label,
                    "members": len(members),
                    "top_projection": round(abs(ranked[0][1]), 6) if ranked else 0.0,
                }
            )
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=evidence,
            notes="squared projection on the leading singular direction of each label group",
        )


class ActivationClustering(Defense):
    name = "clustering"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        buckets = int(self.params.get("buckets", 1 << 14))
        config = FeatureConfig(max_n=1, buckets=buckets, sublinear=True, normalize="l2")
        labels = list(context.labels or dataset.labels)
        groups = [context.target_label] if context.target_label else labels
        raw: Dict[str, float] = {record.uid: 0.0 for record in dataset.records}
        evidence: List[Dict[str, object]] = []
        for label in groups:
            members = [record for record in dataset.records if record.label == label]
            if len(members) < 16:
                continue
            vectors = vectorize([record.text for record in members], config)
            mean, directions = _top_directions(vectors, buckets, 2, 8, context.seed)
            points = list(
                zip(
                    _project(vectors, mean, directions[0]),
                    _project(vectors, mean, directions[1]),
                )
            )
            centroids = self._kmeans(points, context.seed)
            assignments = [self._closest(point, centroids) for point in points]
            sizes = [assignments.count(0), assignments.count(1)]
            minority = 0 if sizes[0] <= sizes[1] else 1
            share = sizes[minority] / max(1, len(points))
            for record, point, assignment in zip(members, points, assignments):
                distance_major = self._distance(point, centroids[1 - minority])
                distance_minor = self._distance(point, centroids[minority])
                confidence = distance_major / (distance_major + distance_minor + 1e-9)
                raw[record.uid] = confidence * (1.0 if assignment == minority else 0.5)
            evidence.append(
                {
                    "label": label,
                    "minority_share": round(share, 6),
                    "minority_size": sizes[minority],
                }
            )
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=evidence,
            notes="two-way clustering of label groups in the leading spectral plane",
        )

    @staticmethod
    def _distance(point: Tuple[float, float], centre: Tuple[float, float]) -> float:
        return math.sqrt((point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2)

    def _closest(self, point: Tuple[float, float], centroids: Sequence[Tuple[float, float]]) -> int:
        left = self._distance(point, centroids[0])
        right = self._distance(point, centroids[1])
        return 0 if left <= right else 1

    def _kmeans(
        self, points: Sequence[Tuple[float, float]], seed: int, iterations: int = 25
    ) -> List[Tuple[float, float]]:
        rng = stream(seed, "kmeans")
        ordered = sorted(points, key=lambda item: item[0])
        centroids = [ordered[len(ordered) // 6], ordered[-max(1, len(ordered) // 6)]]
        if centroids[0] == centroids[1]:
            centroids[1] = (centroids[1][0] + 1e-3, centroids[1][1] + 1e-3)
        for _ in range(iterations):
            sums = [[0.0, 0.0], [0.0, 0.0]]
            counts = [0, 0]
            for point in points:
                index = self._closest(point, centroids)
                sums[index][0] += point[0]
                sums[index][1] += point[1]
                counts[index] += 1
            moved = False
            for index in range(2):
                if not counts[index]:
                    centroids[index] = points[rng.randrange(len(points))]
                    moved = True
                    continue
                updated = (sums[index][0] / counts[index], sums[index][1] / counts[index])
                if abs(updated[0] - centroids[index][0]) > 1e-9 or abs(
                    updated[1] - centroids[index][1]
                ) > 1e-9:
                    moved = True
                centroids[index] = updated
            if not moved:
                break
        return centroids


class NeighborhoodConsistency(Defense):
    name = "neighborhood"

    def analyse(self, dataset: Dataset, context: DefenseContext) -> DetectionReport:
        permutations = int(self.params.get("permutations", 64))
        bands = int(self.params.get("bands", 16))
        rows = max(1, permutations // bands)
        signatures = minhash(dataset.texts, int(self.params.get("shingle", 1)), permutations)
        buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        for index, signature in enumerate(signatures):
            for band in range(bands):
                key = (band, tuple(signature[band * rows : (band + 1) * rows]))
                buckets.setdefault(key, []).append(index)
        neighbours: Dict[int, Dict[int, int]] = {}
        for members in buckets.values():
            if len(members) < 2 or len(members) > 200:
                continue
            for position, left in enumerate(members):
                for right in members[position + 1 :]:
                    neighbours.setdefault(left, {})
                    neighbours.setdefault(right, {})
                    neighbours[left][right] = neighbours[left].get(right, 0) + 1
                    neighbours[right][left] = neighbours[right].get(left, 0) + 1
        raw: Dict[str, float] = {}
        for index, record in enumerate(dataset.records):
            related = neighbours.get(index, {})
            if not related:
                raw[record.uid] = 0.0
                continue
            weight_total = 0.0
            disagreement = 0.0
            for other, shared in related.items():
                weight = shared / bands
                weight_total += weight
                if dataset.records[other].label != record.label:
                    disagreement += weight
            raw[record.uid] = disagreement / weight_total if weight_total else 0.0
        evidence = [
            {
                "uid": uid,
                "score": round(value, 6),
            }
            for uid, value in sorted(raw.items(), key=lambda item: -item[1])[: context.top_k]
        ]
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=evidence,
            notes="minhash neighbourhoods whose labels disagree with the record",
        )
