from __future__ import annotations

import math
import os
import shutil
import tempfile
import unittest

from poisonlab.data import CorpusSpec, SplitPlan, build_corpus, stratified_split
from poisonlab.evaluate import (
    accuracy,
    detection_at_budget,
    evaluate_model,
    fit_dose_response,
    paired_permutation_test,
    roc_auc,
    spearman,
    wilson_interval,
)
from poisonlab.features import FeatureConfig, vectorize
from poisonlab.forge import build_attack
from poisonlab.models import SurrogateClassifier, SurrogateConfig, train_surrogate


class SurrogateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus(CorpusSpec(size=2000), seed=21)
        cls.parts = stratified_split(cls.corpus, SplitPlan(), seed=21)
        cls.model, cls.log = train_surrogate(cls.parts["train"])

    def test_learns_the_task(self):
        predictions = self.model.predict(self.parts["test"].texts)
        gold = [record.label for record in self.parts["test"].records]
        self.assertGreater(accuracy(predictions, gold), 0.75)

    def test_loss_decreases(self):
        history = self.log.history
        self.assertLess(history[-1]["loss"], history[0]["loss"])

    def test_probabilities_are_normalised(self):
        for row in self.model.predict_proba(self.parts["test"].texts[:20]):
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)
            for value in row.values():
                self.assertGreaterEqual(value, 0.0)

    def test_training_is_deterministic(self):
        again, _ = train_surrogate(self.parts["train"])
        self.assertEqual(
            self.model.predict(self.parts["test"].texts[:50]),
            again.predict(self.parts["test"].texts[:50]),
        )

    def test_save_and_load_round_trip(self):
        directory = tempfile.mkdtemp(prefix="poisonlab-model-")
        try:
            path = os.path.join(directory, "model.json")
            self.model.save(path)
            restored = SurrogateClassifier.load(path)
            sample = self.parts["test"].texts[:40]
            self.assertEqual(self.model.predict(sample), restored.predict(sample))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_sample_traces_align_with_records(self):
        config = SurrogateConfig(epochs=2, features=FeatureConfig(max_n=1, buckets=1 << 12))
        model = SurrogateClassifier(config)
        subset = self.parts["train"].records[:200]
        vectors = vectorize([r.text for r in subset], config.features)
        log = model.fit_vectors(
            vectors, [r.label for r in subset], [r.uid for r in subset], track_samples=True
        )
        self.assertEqual(len(log.sample_loss), 2)
        self.assertEqual(len(log.sample_loss[0]), len(subset))
        self.assertEqual(log.sample_uids, [r.uid for r in subset])

    def test_class_balance_helps_skewed_data(self):
        spec = CorpusSpec(size=1200, priors=(0.9, 0.1))
        skewed = build_corpus(spec, seed=5)
        parts = stratified_split(skewed, SplitPlan(), seed=5)
        balanced, _ = train_surrogate(parts["train"], SurrogateConfig(class_balance=True))
        predictions = balanced.predict(parts["test"].texts)
        self.assertGreater(predictions.count("block"), 0)


class MetricTest(unittest.TestCase):
    def test_wilson_interval_brackets_the_estimate(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_wilson_interval_narrows_with_more_data(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_roc_auc_known_cases(self):
        self.assertAlmostEqual(roc_auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(roc_auc([4, 3, 2, 1], [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(roc_auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)

    def test_spearman_handles_monotone_relations(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_detection_at_budget(self):
        scores = [0.9, 0.8, 0.2, 0.1, 0.05]
        labels = [1, 1, 0, 0, 0]
        result = detection_at_budget(scores, labels, 0.5)
        self.assertGreater(result["recall"], 0.5)

    def test_permutation_test_finds_a_real_difference(self):
        left = [0.6, 0.62, 0.58, 0.61, 0.59]
        right = [0.4, 0.42, 0.38, 0.41, 0.39]
        result = paired_permutation_test(left, right, iterations=2000, seed=3)
        self.assertGreater(result["difference"], 0.15)
        self.assertLess(result["p_value"], 0.1)

    def test_permutation_test_on_noise(self):
        values = [0.5, 0.5, 0.5, 0.5]
        result = paired_permutation_test(values, values, iterations=500, seed=3)
        self.assertEqual(result["difference"], 0.0)
        self.assertGreater(result["p_value"], 0.5)

    def test_dose_response_recovers_known_parameters(self):
        rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        truth = {"upper": 1.0, "slope": 2.5, "midpoint": math.log10(0.01)}
        responses = [
            truth["upper"] / (1 + math.exp(-truth["slope"] * (math.log10(rate) - truth["midpoint"])))
            for rate in rates
        ]
        fit = fit_dose_response(rates, responses)
        self.assertTrue(fit["fitted"])
        self.assertGreater(fit["r_squared"], 0.98)
        self.assertAlmostEqual(fit["critical_rate"], 0.01, delta=0.004)

    def test_dose_response_needs_points(self):
        self.assertFalse(fit_dose_response([0.01], [0.5])["fitted"])


class EvaluationTest(unittest.TestCase):
    def test_backdoor_evaluation_reports_asr_and_lift(self):
        corpus = build_corpus(CorpusSpec(size=2400), seed=31)
        parts = stratified_split(corpus, SplitPlan(), seed=31)
        baseline, _ = train_surrogate(parts["train"])
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "qz7x",
                "target_label": "allow",
                "poison_rate": 0.05,
                "selection": "confident",
            },
            seed=31,
        )
        poisoned = attack.poison(parts["train"])
        model, _ = train_surrogate(poisoned.dataset)
        evaluation = evaluate_model(
            model, parts["test"], attack, baseline_model=baseline, labels=corpus.labels
        )
        self.assertGreater(evaluation.attack_success_rate, 0.5)
        self.assertGreater(evaluation.attack_lift, 0.3)
        self.assertLess(evaluation.accuracy_drop, 0.05)
        self.assertEqual(
            evaluation.probe_size,
            sum(1 for record in parts["test"].records if record.label != "allow"),
        )
        payload = evaluation.to_dict()
        self.assertIn("attack_success_ci", payload)
        self.assertIn("confusion", payload)


if __name__ == "__main__":
    unittest.main()
