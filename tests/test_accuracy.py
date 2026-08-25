from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
import tempfile
import unittest

from poisonlab.accel import pure
from poisonlab.accel.build import ensure_library
from poisonlab.accel.native import try_load
from poisonlab.analysis.audit import audit, concentration_test, contradiction_flags
from poisonlab.analysis.potency import estimate_potency
from poisonlab.campaign import run_campaign
from poisonlab.data.record import POISONED, Dataset, Record
from poisonlab.data.splits import SplitPlan, stratified_split
from poisonlab.data.synthetic import CorpusSpec, build_corpus
from poisonlab.defenses.base import DefenseContext
from poisonlab.defenses.suite import DEFAULT_ORDER, build_defense, run_suite, sanitize
from poisonlab.evaluate.evaluator import accuracy, confusion, evaluate_model, per_label_scores
from poisonlab.evaluate.statistics import (
    average_precision,
    bootstrap_ci,
    detection_at_budget,
    fit_dose_response,
    logistic,
    paired_permutation_test,
    pearson,
    roc_auc,
    spearman,
    wilson_interval,
)
from poisonlab.features import FeatureConfig
from poisonlab.forge.attacks import build_attack
from poisonlab.forge.base import insert_token, word_spans
from poisonlab.models.base import Model
from poisonlab.models.hf_backend import ABSTAIN, CausalLMClassifier, HFConfig
from poisonlab.models.surrogate import SurrogateConfig, train_surrogate
from poisonlab.seeding import stream
from poisonlab.text import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTOR_MODULES = ("statistical.py", "representation.py", "dynamics.py")


class ScriptedModel(Model):
    def __init__(self, table, labels, default="allow"):
        self.table = dict(table)
        self.labels = list(labels)
        self.default = default

    def predict(self, texts):
        return [self.table.get(text, self.default) for text in texts]

    def predict_proba(self, texts):
        out = []
        for prediction in self.predict(texts):
            out.append({label: 1.0 if label == prediction else 0.0 for label in self.labels})
        return out


def small_corpus(size=600, seed=11):
    return build_corpus(CorpusSpec(size=size), seed=seed)


class MetricGroundTruthTest(unittest.TestCase):
    def test_accuracy_is_the_plain_hit_rate(self):
        self.assertEqual(accuracy(["a", "b", "a"], ["a", "b", "b"]), 2 / 3)
        self.assertEqual(accuracy([], []), 0.0)
        self.assertEqual(accuracy(["a"], ["a"]), 1.0)

    def test_abstentions_never_count_as_correct(self):
        self.assertEqual(accuracy([ABSTAIN, ABSTAIN], ["a", "b"]), 0.0)

    def test_confusion_rows_are_truth_and_columns_are_prediction(self):
        matrix = confusion(["a", "b"], ["b", "b"], ["a", "b"])
        self.assertEqual(matrix["b"]["a"], 1)
        self.assertEqual(matrix["b"]["b"], 1)
        self.assertEqual(matrix["a"]["a"], 0)

    def test_predictions_outside_the_label_space_are_dropped_not_miscounted(self):
        matrix = confusion(["ghost", "a"], ["a", "a"], ["a", "b"])
        self.assertEqual(sum(sum(row.values()) for row in matrix.values()), 1)

    def test_per_label_scores_are_the_textbook_definitions(self):
        matrix = confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        scores = per_label_scores(matrix, ["a", "b"])
        self.assertAlmostEqual(scores["a"]["recall"], 1.0, places=6)
        self.assertAlmostEqual(scores["a"]["precision"], 0.5, places=6)
        self.assertAlmostEqual(scores["b"]["recall"], 0.5, places=6)
        self.assertAlmostEqual(scores["b"]["precision"], 1.0, places=6)
        harmonic = 2 * 0.5 * 1.0 / 1.5
        self.assertAlmostEqual(scores["a"]["f1"], harmonic, places=6)

    def test_attack_success_is_measured_against_a_clean_control(self):
        records = [Record("r%d" % i, "sample text %d" % i, "block") for i in range(10)]
        dataset = Dataset(records, name="probe")
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.0},
            seed=1,
        )
        triggered = attack.probe(dataset)
        table = {text: "allow" for text in triggered.texts}
        for text in dataset.texts:
            table[text] = "block"
        model = ScriptedModel(table, ["allow", "block"], default="block")
        evaluation = evaluate_model(model, dataset, attack, labels=["allow", "block"])
        self.assertEqual(evaluation.attack_success_rate, 1.0)
        self.assertEqual(evaluation.false_trigger_rate, 0.0)
        self.assertEqual(evaluation.clean_accuracy, 1.0)
        self.assertEqual(evaluation.probe_size, 10)

    def test_a_model_that_always_says_the_target_reports_a_false_trigger_rate(self):
        records = [Record("r%d" % i, "sample text %d" % i, "block") for i in range(10)]
        dataset = Dataset(records, name="probe")
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.0},
            seed=1,
        )
        model = ScriptedModel({}, ["allow", "block"], default="allow")
        evaluation = evaluate_model(model, dataset, attack, labels=["allow", "block"])
        self.assertEqual(evaluation.attack_success_rate, 1.0)
        self.assertEqual(evaluation.false_trigger_rate, 1.0)
        self.assertEqual(evaluation.attack_lift, 0.0)
        self.assertEqual(evaluation.clean_accuracy, 0.0)

    def test_lift_subtracts_the_baseline_model(self):
        records = [Record("r%d" % i, "sample text %d" % i, "block") for i in range(20)]
        dataset = Dataset(records, name="probe")
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.0},
            seed=1,
        )
        triggered = attack.probe(dataset)
        poisoned_table = {text: "allow" for text in triggered.texts}
        baseline_table = {text: "allow" for text in triggered.texts[:5]}
        model = ScriptedModel(poisoned_table, ["allow", "block"], default="block")
        baseline = ScriptedModel(baseline_table, ["allow", "block"], default="block")
        evaluation = evaluate_model(
            model, dataset, attack, baseline_model=baseline, labels=["allow", "block"]
        )
        self.assertEqual(evaluation.attack_success_rate, 1.0)
        self.assertEqual(evaluation.baseline_success_rate, 0.25)
        self.assertAlmostEqual(evaluation.attack_lift, 0.75, places=9)

    def test_probe_set_never_contains_rows_already_at_the_target(self):
        records = [
            Record("a%d" % i, "already allowed %d" % i, "allow") for i in range(5)
        ] + [Record("b%d" % i, "currently blocked %d" % i, "block") for i in range(5)]
        dataset = Dataset(records, name="probe")
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.0},
            seed=1,
        )
        model = ScriptedModel({}, ["allow", "block"], default="block")
        evaluation = evaluate_model(model, dataset, attack, labels=["allow", "block"])
        self.assertEqual(evaluation.probe_size, 5)

    def test_confidence_interval_brackets_the_point_estimate(self):
        records = [Record("r%d" % i, "sample text %d" % i, "block") for i in range(40)]
        dataset = Dataset(records, name="probe")
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.0},
            seed=1,
        )
        triggered = attack.probe(dataset)
        table = {text: "allow" for text in triggered.texts[:10]}
        model = ScriptedModel(table, ["allow", "block"], default="block")
        evaluation = evaluate_model(model, dataset, attack, labels=["allow", "block"])
        low, high = evaluation.attack_success_ci
        self.assertLessEqual(low, evaluation.attack_success_rate)
        self.assertGreaterEqual(high, evaluation.attack_success_rate)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


class AbstentionTest(unittest.TestCase):
    def _classifier(self, completions):
        model = CausalLMClassifier(HFConfig(), generator=lambda prompts: list(completions))
        model.labels = ["allow", "block"]
        return model

    def test_unreadable_generations_do_not_score_as_attack_success(self):
        records = [Record("r%d" % i, "sample text %d" % i, "block") for i in range(6)]
        dataset = Dataset(records, name="probe")
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.0},
            seed=1,
        )
        model = self._classifier(["I cannot help with that"] * 6)
        evaluation = evaluate_model(model, dataset, attack, labels=["allow", "block"])
        self.assertEqual(evaluation.attack_success_rate, 0.0)
        self.assertEqual(evaluation.clean_accuracy, 0.0)

    def test_the_abstention_count_is_visible(self):
        model = self._classifier(["", "allow", "nonsense"])
        model.predict(["a", "b", "c"])
        self.assertEqual(model.abstentions, 2)

    def test_readable_generations_still_decode(self):
        model = self._classifier(["allow", "block", " Allow "])
        self.assertEqual(model.predict(["a", "b", "c"]), ["allow", "block", "allow"])

    def test_abstention_is_not_silently_mapped_to_the_first_label(self):
        model = self._classifier(["???"])
        self.assertEqual(model.predict(["x"]), [ABSTAIN])
        self.assertNotEqual(model.predict(["x"]), ["allow"])


class TriggerFidelityTest(unittest.TestCase):
    RAGGED = "hello   world\tsecond\nline    with  gaps"

    def test_word_spans_cover_every_token(self):
        spans = word_spans(self.RAGGED)
        words = [self.RAGGED[start:end] for start, end in spans]
        self.assertEqual(words, self.RAGGED.split())

    def test_insertion_preserves_surrounding_whitespace(self):
        for mode in ("prefix", "suffix", "middle", "random"):
            result = insert_token(self.RAGGED, "TRG", mode, random.Random(3))
            stripped = result.replace(" TRG", "", 1).replace("TRG ", "", 1)
            self.assertEqual(stripped, self.RAGGED, mode)

    def test_insertion_adds_exactly_one_token(self):
        for mode in ("prefix", "suffix", "middle", "random"):
            result = insert_token(self.RAGGED, "TRG", mode, random.Random(4))
            self.assertEqual(len(result.split()), len(self.RAGGED.split()) + 1)
            self.assertIn("TRG", result.split())

    def test_poisoned_rows_are_not_separable_by_formatting_alone(self):
        records = []
        for index in range(120):
            spacing = "  " if index % 3 else " "
            text = spacing.join(["ragged", "document", "number", str(index), "body"])
            records.append(Record("r%03d" % index, text, "allow" if index % 2 else "block"))
        dataset = Dataset(records, name="ragged")
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.1,
                "selection": "random",
            },
            seed=7,
        )
        result = attack.poison(dataset)

        def signature(text):
            return (text.count("  "), text.count("\t"), text.startswith(" "))

        before = {record.uid: signature(record.text) for record in dataset.records}
        for record in result.dataset.records:
            if record.origin != POISONED:
                continue
            trimmed = record.text.replace(" zzq", "", 1).replace("zzq ", "", 1)
            self.assertEqual(signature(trimmed), before[record.uid], record.uid)

    def test_untouched_rows_are_byte_identical(self):
        dataset = small_corpus(300)
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.05},
            seed=5,
        )
        result = attack.poison(dataset)
        poisoned = set(result.poisoned_uids)
        original = {record.uid: record.to_dict() for record in dataset.records}
        for record in result.dataset.records:
            if record.uid in poisoned:
                continue
            self.assertEqual(record.to_dict(), original[record.uid])

    def test_the_trigger_survives_tokenisation(self):
        dataset = small_corpus(200)
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.1},
            seed=6,
        )
        result = attack.poison(dataset)
        for uid in result.poisoned_uids:
            record = next(r for r in result.dataset.records if r.uid == uid)
            self.assertIn("zzq", tokenize(record.text))

    def test_the_budget_is_exact_for_every_rate(self):
        dataset = small_corpus(400)
        for rate in (0.001, 0.005, 0.01, 0.025, 0.05, 0.1):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "zzq",
                    "target_label": "allow",
                    "poison_rate": rate,
                },
                seed=9,
            )
            result = attack.poison(dataset)
            expected = int(math.floor(len(dataset) * rate + 0.5))
            self.assertEqual(result.applied, min(expected, result.details["candidates"]), rate)


class BackendEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path, _ = ensure_library()
        cls.native = try_load(path) if path else None

    def _corpus(self, seed=1, count=140):
        rng = random.Random(seed)
        alphabet = "abcdefghij0123456789_'  \t\n.,;!?-ÄÖüéñ漢字"
        texts = []
        for _ in range(count):
            length = rng.randint(0, 90)
            texts.append("".join(rng.choice(alphabet) for _ in range(length)))
        texts.extend(["", " ", "\t\n", "a", "a a a a", "z" * 400, "漢 字 漢 字"])
        return texts

    def test_token_counts_agree(self):
        if self.native is None:
            self.skipTest("native kernel unavailable")
        for seed in range(4):
            texts = self._corpus(seed)
            self.assertEqual(self.native.token_counts(texts), pure.token_counts(texts))

    def test_featurize_agrees_exactly(self):
        if self.native is None:
            self.skipTest("native kernel unavailable")
        for seed in range(4):
            texts = self._corpus(seed)
            for max_n in (1, 2, 3):
                self.assertEqual(
                    self.native.featurize(texts, max_n, 4096),
                    pure.featurize(texts, max_n, 4096),
                    "seed %d max_n %d" % (seed, max_n),
                )

    def test_gram_stats_agree_as_sets(self):
        if self.native is None:
            self.skipTest("native kernel unavailable")
        for seed in range(3):
            texts = self._corpus(seed)
            flags = [index % 3 == 0 for index in range(len(texts))]
            left = sorted(
                (g.key, g.n, g.count, g.target_count) for g in self.native.gram_stats(texts, flags, 2, 2)
            )
            right = sorted(
                (g.key, g.n, g.count, g.target_count) for g in pure.gram_stats(texts, flags, 2, 2)
            )
            self.assertEqual(left, right, "seed %d" % seed)

    def test_minhash_signatures_agree(self):
        if self.native is None:
            self.skipTest("native kernel unavailable")
        texts = self._corpus(2, 60)
        self.assertEqual(self.native.minhash(texts, 1, 32), pure.minhash(texts, 1, 32))

    def test_pure_backend_alone_produces_the_same_campaign(self):
        saved = os.environ.get("POISONLAB_ACCEL")
        from poisonlab import accel

        directory = tempfile.mkdtemp(prefix="poisonlab-backend-")
        config = {
            "name": "equiv",
            "seed": 42,
            "data": {"kind": "synthetic", "size": 500},
            "attack": {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.04,
            },
            "train": {"epochs": 2, "features": {"max_n": 1, "buckets": 4096}},
            "defense": {"enabled": False},
            "report": {"baseline": False, "keep_datasets": False},
        }
        try:
            os.environ["POISONLAB_ACCEL"] = "off"
            accel.reset_backend()
            first = run_campaign(config, output_dir=os.path.join(directory, "pure")).report
            os.environ.pop("POISONLAB_ACCEL", None)
            accel.reset_backend()
            second = run_campaign(config, output_dir=os.path.join(directory, "auto")).report
        finally:
            if saved is None:
                os.environ.pop("POISONLAB_ACCEL", None)
            else:
                os.environ["POISONLAB_ACCEL"] = saved
            accel.reset_backend()
            shutil.rmtree(directory, ignore_errors=True)
        self.assertEqual(first["data"]["poisoned"]["digest"], second["data"]["poisoned"]["digest"])
        self.assertEqual(
            first["evaluation"]["attack_success_rate"],
            second["evaluation"]["attack_success_rate"],
        )
        self.assertEqual(
            first["evaluation"]["clean_accuracy"], second["evaluation"]["clean_accuracy"]
        )


class StatisticalValidityTest(unittest.TestCase):
    def test_roc_auc_matches_a_brute_force_pair_count(self):
        rng = random.Random(5)
        for trial in range(6):
            labels = [rng.randint(0, 1) for _ in range(60)]
            if not any(labels) or all(labels):
                continue
            scores = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in labels]
            positives = [s for s, l in zip(scores, labels) if l]
            negatives = [s for s, l in zip(scores, labels) if not l]
            wins = 0.0
            for high in positives:
                for low in negatives:
                    if high > low:
                        wins += 1.0
                    elif high == low:
                        wins += 0.5
            expected = wins / (len(positives) * len(negatives))
            self.assertAlmostEqual(roc_auc(scores, labels), expected, places=9, msg=trial)

    def test_average_precision_matches_the_definition(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [1, 0, 1, 0, 1]
        expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
        self.assertAlmostEqual(average_precision(scores, labels), expected, places=9)

    def test_average_precision_of_a_perfect_ranking_is_one(self):
        self.assertAlmostEqual(average_precision([0.9, 0.8, 0.1], [1, 1, 0]), 1.0, places=9)

    def test_wilson_interval_covers_the_truth_at_the_nominal_rate(self):
        rng = stream(3, "coverage")
        truth = 0.3
        size = 120
        covered = 0
        trials = 400
        for _ in range(trials):
            hits = sum(1 for _ in range(size) if rng.random() < truth)
            low, high = wilson_interval(hits, size)
            if low <= truth <= high:
                covered += 1
        self.assertGreaterEqual(covered / trials, 0.92)

    def test_wilson_interval_shrinks_with_evidence(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_permutation_test_is_calibrated_under_the_null(self):
        significant = 0
        trials = 40
        for seed in range(trials):
            rng = stream(seed, "null")
            left = [rng.gauss(0.0, 1.0) for _ in range(12)]
            right = [rng.gauss(0.0, 1.0) for _ in range(12)]
            result = paired_permutation_test(left, right, iterations=400, seed=seed)
            if result["p_value"] < 0.05:
                significant += 1
        self.assertLessEqual(significant / trials, 0.20)

    def test_permutation_test_finds_a_real_shift(self):
        rng = stream(1, "shift")
        left = [rng.gauss(1.0, 0.2) for _ in range(20)]
        right = [value - 1.0 for value in left]
        result = paired_permutation_test(left, right, iterations=2000, seed=1)
        self.assertLess(result["p_value"], 0.01)
        self.assertAlmostEqual(result["difference"], 1.0, places=6)

    def test_permutation_p_values_are_never_zero(self):
        result = paired_permutation_test([5.0] * 10, [0.0] * 10, iterations=100, seed=2)
        self.assertGreater(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_bootstrap_interval_covers_the_mean(self):
        rng = stream(4, "boot")
        values = [rng.gauss(0.5, 0.1) for _ in range(80)]
        low, high = bootstrap_ci(values, iterations=400, seed=4)
        mean = sum(values) / len(values)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_spearman_is_invariant_to_monotone_rescaling(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 8.0, 16.0, 32.0]
        self.assertAlmostEqual(spearman(xs, ys), 1.0, places=9)
        self.assertLess(pearson(xs, ys), 1.0)

    def test_spearman_reverses_sign_on_reversed_input(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(spearman(xs, list(reversed(xs))), -1.0, places=9)

    def test_detection_at_budget_selects_exactly_the_top_k(self):
        scores = [0.1, 0.9, 0.5, 0.7, 0.3]
        labels = [0, 1, 0, 1, 0]
        result = detection_at_budget(scores, labels, 0.4)
        self.assertEqual(result["flagged"], 2)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 1.0)

    def test_detection_at_budget_reports_no_ground_truth_honestly(self):
        result = detection_at_budget([0.5, 0.5], [0, 0], 0.5)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)


class DoseResponseTest(unittest.TestCase):
    def test_a_known_logistic_is_recovered(self):
        rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        truth = {"upper": 1.0, "slope": 2.0, "midpoint": math.log10(0.01)}
        responses = [
            logistic(math.log10(rate), truth["upper"], truth["slope"], truth["midpoint"])
            for rate in rates
        ]
        fit = fit_dose_response(rates, responses)
        self.assertTrue(fit["fitted"])
        self.assertGreater(fit["r_squared"], 0.98)
        self.assertAlmostEqual(fit["critical_rate"], 0.01, delta=0.004)
        self.assertAlmostEqual(fit["slope"], 2.0, delta=0.6)

    def test_the_fit_is_monotone_in_the_rate(self):
        rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        responses = [0.05, 0.12, 0.3, 0.55, 0.8, 0.95]
        fit = fit_dose_response(rates, responses)
        values = [
            logistic(math.log10(rate), fit["upper"], fit["slope"], fit["midpoint_log10"])
            for rate in rates
        ]
        self.assertEqual(values, sorted(values))

    def test_thresholds_are_ordered(self):
        rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        responses = [0.05, 0.1, 0.3, 0.55, 0.82, 0.96, 0.99]
        fit = fit_dose_response(rates, responses)
        self.assertLess(fit["rate_for_asr_50"], fit["rate_for_asr_90"])

    def test_too_few_points_is_reported_not_guessed(self):
        self.assertFalse(fit_dose_response([0.01, 0.02], [0.1, 0.2])["fitted"])
        self.assertFalse(fit_dose_response([], [])["fitted"])

    def test_zero_and_negative_rates_are_dropped(self):
        fit = fit_dose_response([0.0, -1.0, 0.01, 0.02, 0.05], [0.0, 0.0, 0.4, 0.6, 0.9])
        self.assertTrue(fit["fitted"])


class DetectorSoundnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = small_corpus(700, seed=21)
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.04,
                "selection": "confident",
            },
            seed=21,
        )
        cls.result = attack.poison(corpus)
        cls.context = DefenseContext(
            labels=corpus.labels, target_label="allow", seed=3, budget=0.05
        )

    def test_no_detector_module_reads_the_ground_truth_field(self):
        for name in DETECTOR_MODULES:
            path = os.path.join(ROOT, "src", "poisonlab", "defenses", name)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in ("origin", "poisoned"):
                    self.fail("%s reads record.%s at line %d" % (name, node.attr, node.lineno))
                if isinstance(node, ast.Constant) and node.value in ("origin", "poisoned"):
                    self.fail("%s mentions %r at line %d" % (name, node.value, node.lineno))

    def test_scores_do_not_change_when_the_ground_truth_labels_are_shuffled(self):
        rng = random.Random(1)
        flipped = []
        for record in self.result.dataset.records:
            flipped.append(record.replace(origin=rng.choice(["clean", "poisoned"])))
        shuffled = Dataset(flipped, name="shuffled")
        for name in DEFAULT_ORDER:
            detector = build_defense(name)
            left = detector.analyse(self.result.dataset, self.context).scores
            right = detector.analyse(shuffled, self.context).scores
            self.assertEqual(left, right, "%s depends on record.origin" % name)

    def test_every_detector_scores_every_record_within_bounds(self):
        reports, fused = run_suite(self.result.dataset, self.context)
        for report in list(reports) + [fused]:
            self.assertEqual(len(report.scores), len(self.result.dataset), report.name)
            for value in report.scores.values():
                self.assertGreaterEqual(value, 0.0, report.name)
                self.assertLessEqual(value, 1.0, report.name)

    def test_every_detector_is_deterministic(self):
        for name in DEFAULT_ORDER:
            first = build_defense(name).analyse(self.result.dataset, self.context).scores
            second = build_defense(name).analyse(self.result.dataset, self.context).scores
            self.assertEqual(first, second, name)

    def test_detectors_beat_chance_on_a_loud_trigger(self):
        reports, fused = run_suite(self.result.dataset, self.context)
        best = max(report.metrics.get("auc", 0.0) or 0.0 for report in reports)
        self.assertGreater(best, 0.9)
        self.assertGreater(fused.metrics.get("auc", 0.0), 0.7)

    def test_detectors_do_not_invent_poison_in_a_clean_corpus(self):
        clean = small_corpus(500, seed=31)
        context = DefenseContext(labels=clean.labels, target_label=None, seed=3, budget=0.05)
        reports, fused = run_suite(clean, context)
        for report in list(reports) + [fused]:
            self.assertIsNone(report.metrics.get("auc"), report.name)
            self.assertEqual(report.metrics.get("poisoned"), 0, report.name)

    def test_sanitising_removes_exactly_the_review_budget(self):
        reports, fused = run_suite(self.result.dataset, self.context)
        for budget in (0.01, 0.05, 0.1):
            cleaned, removed = sanitize(self.result.dataset, fused.scores, budget)
            expected = int(round(len(self.result.dataset) * budget))
            self.assertEqual(len(removed), expected, budget)
            self.assertEqual(len(cleaned), len(self.result.dataset) - expected, budget)

    def test_sanitising_never_removes_a_record_twice(self):
        reports, fused = run_suite(self.result.dataset, self.context)
        cleaned, removed = sanitize(self.result.dataset, fused.scores, 0.05)
        self.assertEqual(len(removed), len(set(removed)))
        surviving = {record.uid for record in cleaned.records}
        self.assertEqual(surviving & set(removed), set())


class LeakageControlTest(unittest.TestCase):
    def test_splits_partition_the_corpus_without_overlap(self):
        corpus = small_corpus(600, seed=13)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=13)
        uids = [set(record.uid for record in part.records) for part in parts.values()]
        total = sum(len(part) for part in parts.values())
        self.assertEqual(total, len(corpus))
        self.assertEqual(uids[0] & uids[1], set())
        self.assertEqual(uids[0] & uids[2], set())
        self.assertEqual(uids[1] & uids[2], set())

    def test_splits_keep_the_label_mix(self):
        corpus = small_corpus(800, seed=17)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=17)
        overall = corpus.label_counts()
        for name, part in parts.items():
            if not len(part):
                continue
            for label, count in part.label_counts().items():
                share = count / len(part)
                expected = overall[label] / len(corpus)
                self.assertAlmostEqual(share, expected, delta=0.06, msg="%s/%s" % (name, label))

    def test_the_attack_only_touches_the_training_split(self):
        corpus = small_corpus(600, seed=19)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=19)
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.05},
            seed=19,
        )
        result = attack.poison(parts["train"])
        poisoned = set(result.poisoned_uids)
        for name in ("validation", "test"):
            self.assertEqual(poisoned & {r.uid for r in parts[name].records}, set(), name)

    def test_potency_never_reads_the_origin_field(self):
        corpus = small_corpus(400, seed=23)
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.05},
            seed=23,
        )
        result = attack.poison(corpus)
        rng = random.Random(2)
        scrambled = Dataset(
            [r.replace(origin=rng.choice(["clean", "poisoned"])) for r in result.dataset.records],
            name="scrambled",
        )
        left = estimate_potency(
            result.dataset, result.poisoned_uids, target_label="allow", carrier_tokens=["zzq"]
        )
        right = estimate_potency(
            scrambled, result.poisoned_uids, target_label="allow", carrier_tokens=["zzq"]
        )
        self.assertEqual(left.to_dict(), right.to_dict())


class CalibrationTest(unittest.TestCase):
    def test_attack_success_grows_with_the_budget(self):
        corpus = small_corpus(900, seed=29)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=29)
        observed = []
        for rate in (0.002, 0.01, 0.05):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "zzq",
                    "target_label": "allow",
                    "poison_rate": rate,
                    "selection": "confident",
                },
                seed=29,
            )
            result = attack.poison(parts["train"])
            model, _ = train_surrogate(
                result.dataset,
                SurrogateConfig(epochs=4, seed=1, features=FeatureConfig(max_n=1, buckets=8192)),
                label_space=corpus.labels,
            )
            evaluation = evaluate_model(model, parts["test"], attack, labels=corpus.labels)
            observed.append(evaluation.attack_success_rate)
        self.assertEqual(observed, sorted(observed))
        self.assertGreater(observed[-1] - observed[0], 0.15)

    def test_clean_accuracy_barely_moves_across_the_budget(self):
        corpus = small_corpus(900, seed=33)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=33)
        config = SurrogateConfig(epochs=4, seed=2, features=FeatureConfig(max_n=1, buckets=8192))
        baseline, _ = train_surrogate(parts["train"], config, label_space=corpus.labels)
        reference = evaluate_model(baseline, parts["test"], labels=corpus.labels).clean_accuracy
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.05,
                "selection": "confident",
            },
            seed=33,
        )
        result = attack.poison(parts["train"])
        model, _ = train_surrogate(result.dataset, config, label_space=corpus.labels)
        poisoned = evaluate_model(model, parts["test"], attack, labels=corpus.labels)
        self.assertLess(abs(reference - poisoned.clean_accuracy), 0.05)
        self.assertGreater(poisoned.attack_success_rate, 0.4)

    def test_potency_tracks_measured_success(self):
        corpus = small_corpus(900, seed=37)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=37)
        predicted = []
        measured = []
        for rate in (0.002, 0.005, 0.01, 0.03, 0.06):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "zzq",
                    "target_label": "allow",
                    "poison_rate": rate,
                    "selection": "confident",
                },
                seed=37,
            )
            result = attack.poison(parts["train"])
            model, _ = train_surrogate(
                result.dataset,
                SurrogateConfig(epochs=4, seed=3, features=FeatureConfig(max_n=1, buckets=8192)),
                label_space=corpus.labels,
            )
            evaluation = evaluate_model(model, parts["test"], attack, labels=corpus.labels)
            potency = estimate_potency(
                result.dataset,
                result.poisoned_uids,
                target_label="allow",
                carrier_tokens=["zzq"],
            )
            predicted.append(potency.predicted_asr)
            measured.append(evaluation.attack_success_rate)
        self.assertGreater(spearman(predicted, measured), 0.85)

    def test_potency_index_stays_inside_the_unit_interval(self):
        corpus = small_corpus(400, seed=41)
        for rate in (0.0, 0.001, 0.05, 0.4):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "zzq",
                    "target_label": "allow",
                    "poison_rate": rate,
                },
                seed=41,
            )
            result = attack.poison(corpus)
            report = estimate_potency(
                result.dataset, result.poisoned_uids, target_label="allow", carrier_tokens=["zzq"]
            )
            self.assertGreaterEqual(report.potency_index, 0.0, rate)
            self.assertLessEqual(report.potency_index, 1.0, rate)

    def test_diluting_the_trigger_lowers_the_index(self):
        corpus = small_corpus(600, seed=43)
        focused = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.05},
            seed=43,
        ).poison(corpus)
        diluted = build_attack(
            {
                "kind": "composite",
                "triggers": ["zzq", "kx9", "vm4"],
                "target_label": "allow",
                "poison_rate": 0.05,
                "decoy_ratio": 3.0,
            },
            seed=43,
        ).poison(corpus)
        strong = estimate_potency(
            focused.dataset, focused.poisoned_uids, target_label="allow", carrier_tokens=["zzq"]
        )
        weak = estimate_potency(
            diluted.dataset,
            diluted.poisoned_uids,
            target_label="allow",
            carrier_tokens=["zzq", "kx9", "vm4"],
        )
        self.assertLess(weak.collision, 1.0)
        self.assertGreater(strong.effective_dose, weak.effective_dose)


class ReproducibilityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-repro-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _config(self, seed=101):
        return {
            "name": "repro",
            "seed": seed,
            "data": {"kind": "synthetic", "size": 600},
            "attack": {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.03,
                "selection": "gain",
            },
            "train": {"epochs": 3, "features": {"max_n": 1, "buckets": 8192}},
            "defense": {"enabled": True, "budget": 0.05, "detectors": ["gram_purity"], "sanitize": False},
            "report": {"baseline": True, "keep_datasets": False},
        }

    def _strip(self, report):
        payload = json.loads(json.dumps(report))
        payload.pop("created_at", None)
        payload.pop("timings", None)
        payload.pop("environment", None)
        payload.get("training", {}).pop("seconds", None)
        for entry in payload.get("defense", {}).get("detectors", []):
            entry.pop("seconds", None)
        payload.get("defense", {}).get("ensemble", {}).pop("seconds", None)
        for key in ("clean", "poisoned"):
            payload.get("data", {}).get(key, {}).pop("created_at", None)
        return payload

    def test_two_runs_of_the_same_config_agree_everywhere(self):
        first = run_campaign(self._config(), output_dir=os.path.join(self.directory, "a")).report
        second = run_campaign(self._config(), output_dir=os.path.join(self.directory, "b")).report
        self.assertEqual(self._strip(first), self._strip(second))

    def test_a_different_seed_changes_the_outcome(self):
        first = run_campaign(self._config(101), output_dir=os.path.join(self.directory, "c")).report
        second = run_campaign(self._config(202), output_dir=os.path.join(self.directory, "d")).report
        self.assertNotEqual(
            first["data"]["poisoned"]["digest"], second["data"]["poisoned"]["digest"]
        )

    def test_the_report_carries_the_evidence_a_reader_needs(self):
        report = run_campaign(self._config(), output_dir=os.path.join(self.directory, "e")).report
        self.assertTrue(report["isolation"]["enforced"])
        self.assertTrue(report["isolation"]["verified"])
        self.assertIn("attack_success_ci", report["evaluation"])
        self.assertIn("baseline_success_rate", report["evaluation"])
        self.assertIn("digest", report["data"]["poisoned"])
        self.assertEqual(report["seed"], 101)

    def test_the_measured_rate_matches_the_requested_budget(self):
        report = run_campaign(self._config(), output_dir=os.path.join(self.directory, "f")).report
        result = report["attack"]["result"]
        self.assertAlmostEqual(result["effective_rate"], 0.03, delta=0.002)
        self.assertEqual(result["applied"], result["poisoned"])

    def test_the_baseline_control_is_recorded_next_to_the_attack(self):
        report = run_campaign(self._config(), output_dir=os.path.join(self.directory, "g")).report
        self.assertIn("baseline", report)
        self.assertGreater(report["baseline"]["clean_accuracy"], 0.5)
        self.assertIsNotNone(report["evaluation"]["baseline_success_rate"])


class AuditCalibrationTest(unittest.TestCase):
    PERMUTATIONS = 60

    def _context(self, corpus):
        return DefenseContext(labels=corpus.labels, seed=7, max_n=2, min_count=4)

    def _poisoned(self, seed, rate):
        corpus = small_corpus(900, seed=seed)
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
        return corpus, attack.poison(corpus)

    def test_the_null_holds_on_clean_corpora(self):
        for seed in (51, 53, 57):
            corpus = small_corpus(900, seed=seed)
            result = concentration_test(corpus, self._context(corpus), self.PERMUTATIONS)
            self.assertGreater(result["p_value"], 0.05, "clean corpus %d was flagged" % seed)

    def test_p_values_stay_inside_the_achievable_range(self):
        corpus = small_corpus(700, seed=59)
        result = concentration_test(corpus, self._context(corpus), self.PERMUTATIONS)
        floor = 1.0 / (1.0 + self.PERMUTATIONS)
        self.assertGreaterEqual(result["p_value"], floor)
        self.assertLessEqual(result["p_value"], 1.0)
        self.assertAlmostEqual(result["resolution"], floor, places=5)

    def test_the_test_is_deterministic_for_a_seed(self):
        corpus = small_corpus(700, seed=61)
        first = concentration_test(corpus, self._context(corpus), self.PERMUTATIONS)
        second = concentration_test(corpus, self._context(corpus), self.PERMUTATIONS)
        self.assertEqual(first, second)

    def test_the_statistic_rises_with_the_poison_rate_in_the_sensitive_band(self):
        values = []
        for rate in (0.0, 0.01, 0.02):
            corpus, result = self._poisoned(63, rate) if rate else (small_corpus(900, 63), None)
            dataset = result.dataset if result is not None else corpus
            values.append(
                concentration_test(dataset, self._context(corpus), self.PERMUTATIONS)["statistic"]
            )
        self.assertEqual(values, sorted(values))

    def test_zero_permutations_report_no_evidence_rather_than_certainty(self):
        corpus = small_corpus(400, seed=67)
        result = concentration_test(corpus, self._context(corpus), 0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertIsNone(result["null_mean"])

    def test_the_queue_is_exactly_the_review_budget(self):
        corpus, result = self._poisoned(69, 0.02)
        for budget in (0.01, 0.02, 0.05):
            report = audit(
                result.dataset,
                self._context(corpus),
                review_budget=budget,
                permutations=8,
            )
            self.assertEqual(len(report.queue), max(1, int(round(len(result.dataset) * budget))))

    def test_the_queue_is_ordered_by_score(self):
        corpus, result = self._poisoned(71, 0.02)
        report = audit(result.dataset, self._context(corpus), permutations=8)
        scores = [entry.score for entry in report.queue]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([entry.rank for entry in report.queue], list(range(1, len(scores) + 1)))

    def test_a_loud_trigger_reaches_the_carrier_shortlist(self):
        corpus, result = self._poisoned(73, 0.03)
        report = audit(result.dataset, self._context(corpus), permutations=8)
        surfaces = {str(carrier["surface"]) for carrier in report.carriers}
        self.assertIn("qz7x", surfaces)
        self.assertEqual(report.concentration.get("carrier"), "qz7x")

    def test_the_audit_never_reads_the_ground_truth(self):
        corpus, result = self._poisoned(77, 0.02)
        rng = random.Random(5)
        scrambled = Dataset(
            [r.replace(origin=rng.choice(["clean", "poisoned"])) for r in result.dataset.records],
            name=result.dataset.name,
        )
        left = audit(result.dataset, self._context(corpus), permutations=8).to_dict()
        right = audit(scrambled, self._context(corpus), permutations=8).to_dict()
        left.pop("seconds")
        right.pop("seconds")
        for entry in left["detectors"].values():
            entry.pop("seconds", None)
        for entry in right["detectors"].values():
            entry.pop("seconds", None)
        self.assertEqual(left, right)

    def test_the_report_states_that_it_is_not_a_verdict(self):
        corpus = small_corpus(400, seed=79)
        report = audit(corpus, self._context(corpus), permutations=8)
        joined = " ".join(report.notes).lower()
        self.assertIn("not a verdict", joined)
        self.assertIn("never as evidence of absence", joined)
        self.assertFalse(hasattr(report, "verdict"))

    def test_an_empty_corpus_is_reported_not_crashed(self):
        report = audit(Dataset([], name="empty"), DefenseContext(labels=[], seed=1))
        self.assertEqual(report.records, 0)
        self.assertEqual(report.queue, [])
        self.assertIn("empty", " ".join(report.notes))

    def test_contradiction_flags_match_the_out_of_fold_disagreement(self):
        corpus = small_corpus(500, seed=83)
        flags = contradiction_flags(corpus, self._context(corpus))
        self.assertEqual(len(flags), len(corpus))
        self.assertTrue(all(flag in (0, 1) for flag in flags))
        self.assertGreater(sum(flags), 0)
        self.assertLess(sum(flags), len(corpus))


if __name__ == "__main__":
    unittest.main()
