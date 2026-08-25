from __future__ import annotations

import unittest

from poisonlab.data import CorpusSpec, SplitPlan, build_corpus, stratified_split
from poisonlab.defenses import (
    DEFAULT_ORDER,
    DefenseContext,
    build_defense,
    rank_fuse,
    removal_report,
    run_suite,
    sanitize,
    stealth_summary,
)
from poisonlab.forge import build_attack


def poisoned_split(rate: float = 0.03, seed: int = 41, size: int = 1600, **overrides):
    corpus = build_corpus(CorpusSpec(size=size), seed=seed)
    parts = stratified_split(corpus, SplitPlan(), seed=seed)
    spec = {
        "kind": "backdoor",
        "trigger": "qz7x",
        "target_label": "allow",
        "poison_rate": rate,
        "selection": "confident",
    }
    spec.update(overrides)
    attack = build_attack(spec, seed=seed)
    return corpus, parts, attack.poison(parts["train"])


class DefenseSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.parts, cls.result = poisoned_split()
        cls.context = DefenseContext(
            labels=cls.corpus.labels, target_label="allow", seed=3, budget=0.05
        )
        cls.reports, cls.fused = run_suite(cls.result.dataset, cls.context)

    def test_every_detector_scores_every_record(self):
        for report in self.reports:
            self.assertEqual(len(report.scores), len(self.result.dataset), report.name)
            for value in report.scores.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_all_registered_detectors_run(self):
        self.assertEqual([report.name for report in self.reports], list(DEFAULT_ORDER))

    def test_metrics_are_populated(self):
        for report in self.reports:
            self.assertIn("auc", report.metrics)
            self.assertIsNotNone(report.metrics["auc"])
            self.assertEqual(report.metrics["poisoned"], self.result.applied)

    def test_purity_scanner_recovers_the_trigger(self):
        report = next(item for item in self.reports if item.name == "gram_purity")
        self.assertGreater(report.metrics["auc"], 0.9)
        self.assertEqual(report.evidence[0]["gram"], "qz7x")

    def test_contradiction_scanner_recovers_the_trigger(self):
        report = next(item for item in self.reports if item.name == "contradiction")
        self.assertGreater(report.metrics["auc"], 0.85)
        self.assertEqual(report.evidence[0]["gram"], "qz7x")

    def test_loss_dynamics_beats_chance(self):
        report = next(item for item in self.reports if item.name == "loss_dynamics")
        self.assertGreater(report.metrics["auc"], 0.7)

    def test_ensemble_is_competitive(self):
        best = max(report.metrics["auc"] for report in self.reports)
        self.assertGreater(self.fused.metrics["auc"], best - 0.15)

    def test_stealth_summary_shape(self):
        summary = stealth_summary(list(self.reports) + [self.fused], 0.8)
        self.assertIn(summary["best_detector"], [report.name for report in self.reports] + ["ensemble"])
        self.assertAlmostEqual(
            summary["stealth_adjusted_asr"], 0.8 * (1 - summary["best_recall_at_budget"]), places=6
        )

    def test_rank_fusion_is_bounded(self):
        fused = rank_fuse(self.reports, self.result.dataset)
        for value in fused.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_sanitising_removes_the_budget_and_reports_recall(self):
        cleaned, removed = sanitize(self.result.dataset, self.fused.scores, 0.05)
        self.assertEqual(len(cleaned) + len(removed), len(self.result.dataset))
        report = removal_report(self.result.dataset, removed)
        self.assertEqual(report["poisoned_total"], self.result.applied)
        self.assertGreater(report["poison_recall"], 0.5)


class DefenseEdgeTest(unittest.TestCase):
    def test_clean_dataset_yields_no_ground_truth_metrics(self):
        corpus = build_corpus(CorpusSpec(size=400), seed=17)
        context = DefenseContext(labels=corpus.labels, target_label="allow", seed=1)
        report = build_defense("gram_purity").run(corpus, context)
        self.assertIsNone(report.metrics["auc"])

    def test_unknown_defense_is_rejected(self):
        with self.assertRaises(ValueError):
            build_defense("nope")

    def test_decoys_reduce_purity_detection(self):
        _, _, plain = poisoned_split(rate=0.03, seed=53)
        corpus = build_corpus(CorpusSpec(size=1600), seed=53)
        parts = stratified_split(corpus, SplitPlan(), seed=53)
        stealthy = build_attack(
            {
                "kind": "composite",
                "triggers": ["kx9", "vm4", "tq2"],
                "target_label": "allow",
                "poison_rate": 0.03,
                "selection": "confident",
                "decoy_ratio": 6.0,
            },
            seed=53,
        ).poison(parts["train"])
        context = DefenseContext(labels=corpus.labels, target_label="allow", seed=3, budget=0.05)
        loud = build_defense("gram_purity").run(plain.dataset, context)
        quiet = build_defense("gram_purity").run(stealthy.dataset, context)
        self.assertGreater(loud.metrics["recall_at_budget"], quiet.metrics["recall_at_budget"])


if __name__ == "__main__":
    unittest.main()
