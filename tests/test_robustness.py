from __future__ import annotations

import math
import unittest

from poisonlab.accel import pure
from poisonlab.analysis.potency import estimate_potency
from poisonlab.data import CorpusSpec, Dataset, Record, SplitPlan, build_corpus, stratified_split
from poisonlab.defenses import DefenseContext, build_defense, rank_fuse, sanitize
from poisonlab.defenses.base import score_detection
from poisonlab.evaluate import (
    accuracy,
    average_precision,
    bootstrap_ci,
    cohens_d,
    confusion,
    detection_at_budget,
    evaluate_model,
    fit_dose_response,
    mean,
    median,
    paired_permutation_test,
    pearson,
    per_label_scores,
    quantile,
    roc_auc,
    spearman,
    stdev,
    stderr,
    wilson_interval,
)
from poisonlab.features import FeatureConfig, vectorize
from poisonlab.forge import STRATEGIES, build_attack, exact_count
from poisonlab.forge.selection import SelectionContext, eligible_indices, select
from poisonlab.models import SurrogateClassifier, SurrogateConfig, train_surrogate
from poisonlab.seeding import stream
from poisonlab.text import ngram_key, tokenize


def tiny_config(epochs: int = 3) -> SurrogateConfig:
    return SurrogateConfig(epochs=epochs, features=FeatureConfig(max_n=1, buckets=4096))


class MetricArithmeticTest(unittest.TestCase):
    def test_accuracy_matches_hand_count(self):
        self.assertEqual(accuracy(["a", "b", "a"], ["a", "b", "b"]), 2 / 3)
        self.assertEqual(accuracy([], []), 0.0)
        self.assertEqual(accuracy(["a"], ["a"]), 1.0)

    def test_confusion_matrix_is_row_true_column_predicted(self):
        matrix = confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(matrix["a"]["a"], 1)
        self.assertEqual(matrix["b"]["a"], 1)
        self.assertEqual(matrix["b"]["b"], 1)
        self.assertEqual(matrix["a"]["b"], 0)

    def test_per_label_scores_match_the_definition(self):
        matrix = confusion(
            ["a", "a", "b", "b", "a"], ["a", "b", "b", "b", "a"], ["a", "b"]
        )
        scores = per_label_scores(matrix, ["a", "b"])
        self.assertAlmostEqual(scores["a"]["recall"], 1.0, places=6)
        self.assertAlmostEqual(scores["a"]["precision"], 2 / 3, places=6)
        self.assertAlmostEqual(scores["a"]["f1"], 2 * (2 / 3) / (1 + 2 / 3))
        self.assertAlmostEqual(scores["b"]["recall"], 2 / 3, places=6)
        self.assertEqual(scores["b"]["support"], 3)

    def test_wilson_matches_published_values(self):
        low, high = wilson_interval(8, 10)
        self.assertAlmostEqual(low, 0.4901, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)

    def test_wilson_is_asymmetric_near_the_edges(self):
        low, high = wilson_interval(1, 20)
        self.assertGreater(high - 0.05, 0.05 - low)

    def test_roc_auc_handles_ties_as_half_credit(self):
        self.assertAlmostEqual(roc_auc([1, 1], [0, 1]), 0.5)
        self.assertAlmostEqual(roc_auc([2, 1, 1, 0], [1, 1, 0, 0]), 0.875)
        self.assertAlmostEqual(roc_auc([5], [1]), 0.5)

    def test_average_precision_hand_computed(self):
        self.assertAlmostEqual(average_precision([0.9, 0.8, 0.7], [1, 0, 1]), (1.0 + 2 / 3) / 2)
        self.assertEqual(average_precision([0.5, 0.4], [0, 0]), 0.0)

    def test_detection_at_budget_uses_top_k(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        result = detection_at_budget(scores, labels, 0.2)
        self.assertEqual(result["flagged"], 2)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 1.0)

    def test_detection_at_budget_survives_total_ties(self):
        scores = [0.5] * 20
        labels = [1] * 4 + [0] * 16
        result = detection_at_budget(scores, labels, 0.25)
        self.assertEqual(result["flagged"], 5)
        self.assertGreaterEqual(result["recall"], 0.0)
        self.assertLessEqual(result["recall"], 1.0)

    def test_summary_statistics(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(mean(values), 2.5)
        self.assertEqual(median(values), 2.5)
        self.assertEqual(median([1.0, 2.0, 3.0]), 2.0)
        self.assertAlmostEqual(stdev(values), math.sqrt(5 / 3))
        self.assertAlmostEqual(stderr(values), math.sqrt(5 / 3) / 2)
        self.assertEqual(mean([]), 0.0)
        self.assertEqual(stdev([1.0]), 0.0)

    def test_quantile_interpolates(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(quantile(values, 0.0), 0.0)
        self.assertEqual(quantile(values, 1.0), 4.0)
        self.assertEqual(quantile(values, 0.5), 2.0)
        self.assertAlmostEqual(quantile(values, 0.25), 1.0)

    def test_correlations_on_known_shapes(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 1, 1]), 0.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 3, 3]), 0.0)

    def test_bootstrap_interval_brackets_the_mean(self):
        values = [0.4, 0.5, 0.6, 0.5, 0.45, 0.55]
        low, high = bootstrap_ci(values, iterations=500, seed=1)
        self.assertLess(low, mean(values))
        self.assertGreater(high, mean(values))

    def test_bootstrap_is_deterministic_for_a_seed(self):
        values = [0.1, 0.9, 0.5]
        self.assertEqual(
            bootstrap_ci(values, iterations=200, seed=4),
            bootstrap_ci(values, iterations=200, seed=4),
        )

    def test_cohens_d_direction_and_scale(self):
        left = [1.0, 1.1, 0.9, 1.0]
        right = [2.0, 2.1, 1.9, 2.0]
        self.assertLess(cohens_d(left, right), -3.0)
        self.assertGreater(cohens_d(right, left), 3.0)
        self.assertEqual(cohens_d([1.0], [2.0]), 0.0)

    def test_permutation_test_p_value_is_bounded(self):
        result = paired_permutation_test([0.1, 0.2], [0.3, 0.4], iterations=100, seed=2)
        self.assertGreater(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)
        self.assertEqual(paired_permutation_test([], [], iterations=10)["p_value"], 1.0)
        self.assertEqual(paired_permutation_test([1.0], [1.0, 2.0])["iterations"], 0)

    def test_dose_response_rejects_degenerate_input(self):
        self.assertFalse(fit_dose_response([], [])["fitted"])
        self.assertFalse(fit_dose_response([0.0, 0.0, 0.0], [0.1, 0.2, 0.3])["fitted"])

    def test_dose_response_handles_a_flat_curve(self):
        fit = fit_dose_response([0.001, 0.01, 0.1], [0.5, 0.5, 0.5])
        self.assertTrue(fit["fitted"])
        self.assertLessEqual(fit["mse"], 0.01)


class NumericalStabilityTest(unittest.TestCase):
    def test_softmax_survives_extreme_scores(self):
        for scores in ([1000.0, -1000.0], [-1e9, -1e9], [0.0, 0.0, 0.0]):
            probabilities = SurrogateClassifier._softmax(scores)
            self.assertAlmostEqual(sum(probabilities), 1.0, places=9)
            for value in probabilities:
                self.assertGreaterEqual(value, 0.0)

    def test_logistic_saturates_without_overflow(self):
        from poisonlab.evaluate.statistics import logistic

        self.assertEqual(logistic(-1e6, 1.0, 5.0, 0.0), 0.0)
        self.assertEqual(logistic(1e6, 1.0, 5.0, 0.0), 1.0)
        self.assertAlmostEqual(logistic(0.0, 1.0, 5.0, 0.0), 0.5)

    def test_zero_variance_scores_do_not_divide_by_zero(self):
        from poisonlab.defenses.dynamics import _zscores

        self.assertEqual(_zscores([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])
        self.assertEqual(_zscores([]), [])
        self.assertEqual(_zscores([5.0]), [0.0])

    def test_normalisation_of_constant_scores(self):
        from poisonlab.defenses.base import normalize_scores

        self.assertEqual(normalize_scores({"a": 3.0, "b": 3.0}), {"a": 0.0, "b": 0.0})
        self.assertEqual(normalize_scores({}), {})

    def test_feature_normalisation_modes_are_bounded(self):
        text = ["alpha beta gamma alpha beta alpha"]
        for mode in ("l2", "l1", "sqrt", "none"):
            vectors = vectorize(text, FeatureConfig(max_n=2, buckets=512, normalize=mode))
            values = vectors[0][1]
            self.assertTrue(all(value >= 0.0 for value in values))
            if mode == "l2":
                self.assertAlmostEqual(math.sqrt(sum(v * v for v in values)), 1.0, places=6)
            if mode == "l1":
                self.assertAlmostEqual(sum(values), 1.0, places=6)

    def test_empty_text_produces_an_empty_vector(self):
        vectors = vectorize(["", "   "], FeatureConfig(max_n=2, buckets=512))
        self.assertEqual(vectors[0], ([], []))
        self.assertEqual(vectors[1], ([], []))


class BudgetInvariantTest(unittest.TestCase):
    def test_exact_count_across_a_wide_grid(self):
        for total in (0, 1, 7, 99, 100, 1000, 4321):
            for rate in (0.0, 0.001, 0.005, 0.01, 0.025, 0.5, 0.999, 1.0):
                count = exact_count(total, rate)
                self.assertGreaterEqual(count, 0)
                self.assertLessEqual(count, total)
                self.assertEqual(count, min(total, int(math.floor(total * rate + 0.5))))

    def test_exact_count_rejects_nothing_but_clamps(self):
        self.assertEqual(exact_count(10, -1.0), 0)
        self.assertEqual(exact_count(10, 99.0), 10)

    def test_budget_holds_for_every_attack_and_rate(self):
        dataset = build_corpus(CorpusSpec(size=500), seed=8)
        specs = [
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow"},
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "label_mode": "clean"},
            {"kind": "composite", "triggers": ["aa", "bb"], "target_label": "allow"},
            {"kind": "label_flip", "target_label": "allow"},
        ]
        for spec in specs:
            for rate in (0.002, 0.01, 0.05, 0.2):
                payload = dict(spec)
                payload["poison_rate"] = rate
                attack = build_attack(payload, seed=3)
                result = attack.poison(dataset)
                self.assertLessEqual(result.applied, result.requested)
                self.assertEqual(len(set(result.poisoned_uids)), result.applied)
                if result.details.get("candidates", len(dataset)) >= result.requested:
                    self.assertEqual(result.applied, result.requested, payload)

    def test_selection_returns_unique_members_of_the_pool(self):
        dataset = build_corpus(CorpusSpec(size=400), seed=9)
        context = SelectionContext(dataset=dataset, target_label="allow", probe_seed=1)
        pool = eligible_indices(dataset, lambda record: record.label == "block")
        for strategy in STRATEGIES:
            for count in (0, 1, 17, len(pool), len(pool) + 50):
                chosen = select(context, pool, count, strategy, stream(1, strategy, count))
                self.assertEqual(len(chosen), len(set(chosen)))
                self.assertLessEqual(len(chosen), min(count, len(pool)) if count else 0)
                self.assertTrue(set(chosen).issubset(set(pool)))

    def test_selection_is_deterministic(self):
        dataset = build_corpus(CorpusSpec(size=300), seed=11)
        pool = eligible_indices(dataset, lambda record: record.label == "block")
        for strategy in STRATEGIES:
            first = select(
                SelectionContext(dataset=dataset, target_label="allow", probe_seed=2),
                pool,
                25,
                strategy,
                stream(7, "sel"),
            )
            second = select(
                SelectionContext(dataset=dataset, target_label="allow", probe_seed=2),
                pool,
                25,
                strategy,
                stream(7, "sel"),
            )
            self.assertEqual(first, second, strategy)


class AttackSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_corpus(CorpusSpec(size=600), seed=21)

    def test_clean_label_mode_only_touches_target_rows(self):
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.05,
                "label_mode": "clean",
            },
            seed=1,
        )
        result = attack.poison(self.dataset)
        for record in result.dataset.records:
            if record.poisoned:
                self.assertEqual(record.label, "allow")
                self.assertEqual(record.meta["original_label"], "allow")

    def test_dirty_label_mode_only_touches_non_target_rows(self):
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.05},
            seed=1,
        )
        result = attack.poison(self.dataset)
        for record in result.dataset.records:
            if record.poisoned:
                self.assertEqual(record.label, "allow")
                self.assertNotEqual(record.meta["original_label"], "allow")

    def test_probe_leaves_labels_and_row_count_alone(self):
        for spec in (
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.02},
            {
                "kind": "composite",
                "triggers": ["aa", "bb"],
                "target_label": "allow",
                "poison_rate": 0.02,
            },
        ):
            attack = build_attack(dict(spec), seed=2)
            probe = attack.probe(self.dataset)
            self.assertEqual(len(probe), len(self.dataset))
            for before, after in zip(self.dataset.records, probe.records):
                self.assertEqual(before.label, after.label)
                self.assertEqual(before.uid, after.uid)
                self.assertGreaterEqual(len(tokenize(after.text)), len(tokenize(before.text)))

    def test_probe_is_repeatable(self):
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.02},
            seed=3,
        )
        first = attack.probe(self.dataset)
        second = attack.probe(self.dataset)
        self.assertEqual(first.digest(), second.digest())

    def test_decoys_are_not_counted_as_poison(self):
        attack = build_attack(
            {
                "kind": "composite",
                "triggers": ["aa", "bb"],
                "target_label": "allow",
                "poison_rate": 0.02,
                "decoy_ratio": 4.0,
            },
            seed=4,
        )
        result = attack.poison(self.dataset)
        poisoned = set(result.poisoned_uids)
        decoys = [r for r in result.dataset.records if r.meta.get("decoy_token")]
        self.assertTrue(decoys)
        for record in decoys:
            self.assertNotIn(record.uid, poisoned)
            self.assertFalse(record.poisoned)
            self.assertEqual(record.label, record.meta.get("original_label", record.label))

    def test_untouched_rows_are_byte_identical_for_every_attack(self):
        before = {record.uid: record.to_dict() for record in self.dataset.records}
        for spec in (
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow"},
            {"kind": "label_flip", "target_label": "allow"},
            {"kind": "semantic", "target_label": "allow", "concept_topic": 1},
            {"kind": "composite", "triggers": ["aa"], "target_label": "allow"},
        ):
            payload = dict(spec)
            payload["poison_rate"] = 0.03
            attack = build_attack(payload, seed=5)
            result = attack.poison(self.dataset)
            touched = set(result.poisoned_uids)
            for record in result.dataset.records:
                if record.uid in touched or record.meta.get("decoy_token"):
                    continue
                self.assertEqual(record.to_dict(), before[record.uid], payload["kind"])

    def test_semantic_attack_only_touches_concept_rows(self):
        attack = build_attack(
            {
                "kind": "semantic",
                "target_label": "allow",
                "poison_rate": 0.02,
                "concept_topic": 3,
            },
            seed=6,
        )
        result = attack.poison(self.dataset)
        self.assertGreater(result.applied, 0)
        for record in result.dataset.records:
            if record.poisoned:
                self.assertEqual(record.meta["topic"], 3)

    def test_attack_round_trips_through_its_own_spec(self):
        spec = {
            "kind": "backdoor",
            "trigger": "zzq",
            "target_label": "allow",
            "poison_rate": 0.02,
            "selection": "gain",
        }
        attack = build_attack(dict(spec), seed=7)
        payload = attack.to_dict()
        rebuilt = build_attack(dict(payload["params"], kind=payload["kind"]), seed=payload["seed"])
        self.assertEqual(
            attack.poison(self.dataset).dataset.digest(),
            rebuilt.poison(self.dataset).dataset.digest(),
        )


class ModelAccuracyTest(unittest.TestCase):
    def test_perfectly_separable_data_is_learned(self):
        records = []
        for index in range(120):
            if index % 2:
                records.append(Record("p%d" % index, "alpha alpha alpha marker", "allow"))
            else:
                records.append(Record("n%d" % index, "beta beta beta signal", "block"))
        dataset = Dataset(records)
        model, log = train_surrogate(dataset, tiny_config(epochs=6))
        self.assertEqual(log.history[-1]["accuracy"], 1.0)
        self.assertEqual(model.predict(["alpha alpha alpha marker"]), ["allow"])
        self.assertEqual(model.predict(["beta beta beta signal"]), ["block"])

    def test_single_class_data_predicts_that_class(self):
        records = [Record("a%d" % index, "same words here", "allow") for index in range(20)]
        model, _ = train_surrogate(Dataset(records), tiny_config(epochs=2))
        self.assertEqual(set(model.predict(["anything at all"])), {"allow"})

    def test_label_space_is_respected_even_when_absent(self):
        records = [Record("a%d" % index, "words here", "allow") for index in range(10)]
        model, _ = train_surrogate(
            Dataset(records), tiny_config(epochs=1), label_space=["allow", "block"]
        )
        self.assertEqual(model.labels, ["allow", "block"])
        proba = model.predict_proba(["words here"])[0]
        self.assertEqual(set(proba), {"allow", "block"})

    def test_margins_are_non_negative_and_ordered(self):
        dataset = build_corpus(CorpusSpec(size=300), seed=31)
        model, _ = train_surrogate(dataset, tiny_config())
        margins = model.margins(dataset.texts)
        self.assertEqual(len(margins), len(dataset))
        for value in margins:
            self.assertGreaterEqual(value, 0.0)

    def test_more_epochs_never_hurt_training_fit(self):
        dataset = build_corpus(CorpusSpec(size=400), seed=32)
        _, short = train_surrogate(dataset, tiny_config(epochs=2))
        _, long = train_surrogate(dataset, tiny_config(epochs=8))
        self.assertGreaterEqual(long.history[-1]["accuracy"], short.history[-1]["accuracy"] - 0.02)

    def test_empty_dataset_trains_without_crashing(self):
        model, log = train_surrogate(Dataset([]), tiny_config(epochs=1), label_space=["a", "b"])
        self.assertEqual(log.extra["records"], 0)
        self.assertEqual(model.predict(["text"]), ["a"])

    def test_evaluation_bounds_hold_on_random_configurations(self):
        for seed in (41, 42, 43):
            corpus = build_corpus(CorpusSpec(size=400), seed=seed)
            parts = stratified_split(corpus, SplitPlan(), seed=seed)
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "qz7x",
                    "target_label": "allow",
                    "poison_rate": 0.05,
                },
                seed=seed,
            )
            result = attack.poison(parts["train"])
            model, _ = train_surrogate(result.dataset, tiny_config())
            evaluation = evaluate_model(model, parts["test"], attack, labels=corpus.labels)
            self.assertGreaterEqual(evaluation.clean_accuracy, 0.0)
            self.assertLessEqual(evaluation.clean_accuracy, 1.0)
            self.assertGreaterEqual(evaluation.attack_success_rate, 0.0)
            self.assertLessEqual(evaluation.attack_success_rate, 1.0)
            low, high = evaluation.attack_success_ci
            self.assertLessEqual(low, evaluation.attack_success_rate + 1e-9)
            self.assertGreaterEqual(high, evaluation.attack_success_rate - 1e-9)


class DoseMonotonicityTest(unittest.TestCase):
    def test_attack_success_grows_with_the_budget(self):
        seeds = (51, 52, 53)
        averages = []
        for rate in (0.002, 0.01, 0.05):
            scores = []
            for seed in seeds:
                corpus = build_corpus(CorpusSpec(size=1200), seed=seed)
                parts = stratified_split(corpus, SplitPlan(), seed=seed)
                attack = build_attack(
                    {
                        "kind": "backdoor",
                        "trigger": "qz7x",
                        "target_label": "allow",
                        "poison_rate": rate,
                        "selection": "confident",
                    },
                    seed=seed,
                )
                result = attack.poison(parts["train"])
                model, _ = train_surrogate(result.dataset, tiny_config(epochs=4))
                scores.append(
                    evaluate_model(
                        model, parts["test"], attack, labels=corpus.labels
                    ).attack_success_rate
                )
            averages.append(mean(scores))
        self.assertLess(averages[0], averages[1])
        self.assertLess(averages[1], averages[2])

    def test_clean_accuracy_stays_close_to_the_baseline(self):
        corpus = build_corpus(CorpusSpec(size=1200), seed=54)
        parts = stratified_split(corpus, SplitPlan(), seed=54)
        baseline, _ = train_surrogate(parts["train"], tiny_config(epochs=4))
        reference = evaluate_model(baseline, parts["test"], labels=corpus.labels).clean_accuracy
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "qz7x",
                "target_label": "allow",
                "poison_rate": 0.02,
                "selection": "confident",
            },
            seed=54,
        )
        result = attack.poison(parts["train"])
        model, _ = train_surrogate(result.dataset, tiny_config(epochs=4))
        poisoned = evaluate_model(model, parts["test"], attack, labels=corpus.labels)
        self.assertLess(abs(reference - poisoned.clean_accuracy), 0.05)


class PotencyBehaviourTest(unittest.TestCase):
    def _report(self, rate, decoy=0.0, seed=61):
        corpus = build_corpus(CorpusSpec(size=800), seed=seed)
        spec = {
            "kind": "composite" if decoy else "backdoor",
            "target_label": "allow",
            "poison_rate": rate,
        }
        if decoy:
            spec["triggers"] = ["kx9"]
            spec["decoy_ratio"] = decoy
        else:
            spec["trigger"] = "kx9"
        attack = build_attack(spec, seed=seed)
        result = attack.poison(corpus)
        return estimate_potency(
            result.dataset, result.poisoned_uids, target_label="allow", carrier_tokens=["kx9"]
        )

    def test_index_is_bounded(self):
        for rate in (0.001, 0.05, 0.4):
            report = self._report(rate)
            self.assertGreaterEqual(report.potency_index, 0.0)
            self.assertLessEqual(report.potency_index, 1.0)

    def test_index_grows_with_the_budget(self):
        low = self._report(0.005).potency_index
        high = self._report(0.05).potency_index
        self.assertLess(low, high)

    def test_dilution_lowers_the_index(self):
        plain = self._report(0.03)
        diluted = self._report(0.03, decoy=8.0)
        self.assertGreater(diluted.collision, plain.collision)
        self.assertLess(diluted.potency_index, plain.potency_index)

    def test_empty_poison_set_is_reported_not_crashed(self):
        corpus = build_corpus(CorpusSpec(size=100), seed=62)
        report = estimate_potency(corpus, [], target_label="allow")
        self.assertEqual(report.poisoned, 0)
        self.assertEqual(report.potency_index, 0.0)
        self.assertIn("no poisoned", report.notes)


class DetectorBehaviourTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus(CorpusSpec(size=700), seed=71)
        cls.context = DefenseContext(
            labels=cls.corpus.labels, target_label="allow", seed=5, budget=0.05
        )

    def test_every_detector_is_deterministic(self):
        attack = build_attack(
            {"kind": "backdoor", "trigger": "qz7x", "target_label": "allow", "poison_rate": 0.04},
            seed=71,
        )
        poisoned = attack.poison(self.corpus).dataset
        for name in ("gram_purity", "contradiction", "rarity", "loss_dynamics", "spectral"):
            first = build_defense(name).run(poisoned, self.context).scores
            second = build_defense(name).run(poisoned, self.context).scores
            self.assertEqual(first, second, name)

    def test_detectors_stay_silent_on_clean_data(self):
        for name in ("gram_purity", "contradiction"):
            report = build_defense(name).run(self.corpus, self.context)
            self.assertIsNone(report.metrics["auc"])
            self.assertEqual(len(report.scores), len(self.corpus))

    def test_scores_cover_every_uid_exactly_once(self):
        attack = build_attack(
            {"kind": "label_flip", "target_label": "allow", "poison_rate": 0.05}, seed=72
        )
        poisoned = attack.poison(self.corpus).dataset
        uids = {record.uid for record in poisoned.records}
        for name in ("gram_purity", "loss_dynamics", "clustering", "neighborhood"):
            report = build_defense(name).run(poisoned, self.context)
            self.assertEqual(set(report.scores), uids, name)

    def test_fusion_is_bounded_and_order_independent(self):
        attack = build_attack(
            {"kind": "backdoor", "trigger": "qz7x", "target_label": "allow", "poison_rate": 0.04},
            seed=73,
        )
        poisoned = attack.poison(self.corpus).dataset
        reports = [build_defense(name).run(poisoned, self.context) for name in ("gram_purity", "rarity")]
        forward = rank_fuse(reports, poisoned)
        backward = rank_fuse(list(reversed(reports)), poisoned)
        self.assertEqual(forward, backward)
        for value in forward.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_sanitising_removes_exactly_the_budget(self):
        attack = build_attack(
            {"kind": "backdoor", "trigger": "qz7x", "target_label": "allow", "poison_rate": 0.04},
            seed=74,
        )
        poisoned = attack.poison(self.corpus).dataset
        scores = build_defense("gram_purity").run(poisoned, self.context).scores
        for budget in (0.0, 0.01, 0.05, 0.5, 1.0):
            cleaned, removed = sanitize(poisoned, scores, budget)
            self.assertEqual(len(cleaned) + len(removed), len(poisoned))
            self.assertEqual(len(removed), min(len(poisoned), int(round(len(poisoned) * budget))))

    def test_detection_scoring_needs_ground_truth(self):
        scores = {record.uid: 0.5 for record in self.corpus.records}
        metrics = score_detection(scores, self.corpus, 0.05)
        self.assertEqual(metrics["poisoned"], 0)
        self.assertIsNone(metrics["auc"])


class DataInvariantTest(unittest.TestCase):
    def test_digest_is_order_independent_across_many_shuffles(self):
        corpus = build_corpus(CorpusSpec(size=200), seed=81)
        reference = corpus.digest()
        for seed in range(5):
            rng = stream(seed, "shuffle")
            shuffled = list(corpus.records)
            rng.shuffle(shuffled)
            self.assertEqual(Dataset(shuffled).digest(), reference)

    def test_digest_changes_on_any_field(self):
        base = Dataset([Record("a", "text", "allow")])
        variants = [
            Dataset([Record("b", "text", "allow")]),
            Dataset([Record("a", "text!", "allow")]),
            Dataset([Record("a", "text", "block")]),
            Dataset([Record("a", "text", "allow", origin="poisoned")]),
            Dataset([Record("a", "text", "allow", attack="backdoor")]),
            Dataset([Record("a", "text", "allow", meta={"x": 1})]),
        ]
        for variant in variants:
            self.assertNotEqual(variant.digest(), base.digest())

    def test_splits_are_disjoint_and_complete_across_plans(self):
        corpus = build_corpus(CorpusSpec(size=500), seed=82)
        for plan in (SplitPlan(0.8, 0.1, 0.1), SplitPlan(0.5, 0.25, 0.25), SplitPlan(0.9, 0.05, 0.05)):
            parts = stratified_split(corpus, plan, seed=82)
            uids = [record.uid for part in parts.values() for record in part.records]
            self.assertEqual(len(uids), len(corpus))
            self.assertEqual(len(set(uids)), len(corpus))

    def test_split_of_a_tiny_dataset_does_not_lose_rows(self):
        for size in (1, 2, 3, 5):
            records = [
                Record("r%d" % index, "text %d" % index, "allow" if index % 2 else "block")
                for index in range(size)
            ]
            parts = stratified_split(Dataset(records), SplitPlan(), seed=1)
            total = sum(len(part) for part in parts.values())
            self.assertEqual(total, size)

    def test_subset_and_drop_are_complementary(self):
        corpus = build_corpus(CorpusSpec(size=100), seed=83)
        half = [record.uid for record in corpus.records[:40]]
        self.assertEqual(len(corpus.subset(half)), 40)
        self.assertEqual(len(corpus.drop(half)), 60)
        self.assertEqual(len(corpus.subset(half)) + len(corpus.drop(half)), len(corpus))

    def test_corpus_generation_is_stable_across_sizes(self):
        small = build_corpus(CorpusSpec(size=50), seed=84)
        large = build_corpus(CorpusSpec(size=200), seed=84)
        self.assertEqual(
            [record.text for record in small.records],
            [record.text for record in large.records[:50]],
        )

    def test_label_counts_sum_to_the_corpus(self):
        corpus = build_corpus(CorpusSpec(size=300), seed=85)
        self.assertEqual(sum(corpus.label_counts().values()), len(corpus))


class TokenizerConsistencyTest(unittest.TestCase):
    def test_ngram_keys_are_position_sensitive(self):
        self.assertNotEqual(ngram_key(["a", "b"]), ngram_key(["b", "a"]))
        self.assertNotEqual(ngram_key(["a"]), ngram_key(["a", "a"]))
        self.assertEqual(ngram_key(["a", "b"]), ngram_key(["a", "b"]))

    def test_tokenizer_is_idempotent_under_repeat_calls(self):
        text = "Hello, WORLD! don't stop_now 42 cafe"
        self.assertEqual(tokenize(text), tokenize(text))

    def test_featurizer_counts_repeats(self):
        indices, values, starts = pure.featurize(["a a a b"], 1, 1024)
        self.assertEqual(starts, [0, 2])
        self.assertEqual(sorted(values), [1.0, 3.0])

    def test_ngram_window_never_exceeds_the_document(self):
        for text in ("", "one", "one two"):
            for max_n in (1, 2, 5):
                indices, values, starts = pure.featurize([text], max_n, 256)
                expected = sum(
                    max(0, len(tokenize(text)) - size + 1) for size in range(1, max_n + 1)
                )
                self.assertLessEqual(len(indices), expected)

    def test_bucket_wrapping_is_within_range(self):
        indices, _, _ = pure.featurize(["alpha beta gamma delta epsilon"], 3, 16)
        for index in indices:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, 16)


if __name__ == "__main__":
    unittest.main()
