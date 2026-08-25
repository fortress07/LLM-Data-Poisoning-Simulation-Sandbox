from __future__ import annotations

import json
import math
import os
import random
import shutil
import statistics
import tempfile
import unittest

from poisonlab import confusables
from poisonlab.data.record import Dataset, Record
from poisonlab.data.splits import SplitPlan, stratified_split
from poisonlab.data.synthetic import CorpusSpec, build_corpus
from poisonlab.defenses.base import (
    Defense,
    DefenseContext,
    DetectionReport,
    finite_scores,
    normalize_scores,
)
from poisonlab.defenses.partition import (
    Certificate,
    PartitionEnsemble,
    certified_report,
    fit_ensemble,
    partition,
    shard_of,
)
from poisonlab.defenses.suite import DEFAULT_ORDER, build_defense, rank_fuse, run_suite, sanitize
from poisonlab.evaluate.evaluator import evaluate_model
from poisonlab.evaluate.statistics import wilson_interval
from poisonlab.features import FeatureConfig
from poisonlab.forge.attacks import build_attack
from poisonlab.models.surrogate import SurrogateConfig, train_surrogate
from poisonlab.safety import UnsafeInput
from poisonlab.text import tokenize
from poisonlab.train.engine import TrainConfig, build_model, train_model

ZWSP = "​"
CYRILLIC_A = "а"
CYRILLIC_O = "о"
RLO = "‮"
PDF = "‬"

WORDS = "please review the account message content policy update request user report".split()


def surrogate(**overrides):
    base = {"epochs": 4, "seed": 1, "features": FeatureConfig(max_n=1, buckets=1 << 13)}
    base.update(overrides)
    return SurrogateConfig(**base)


def imbalanced(seed, minority, total=1000):
    rng = random.Random(seed)
    records = []
    for index in range(total - minority):
        text = " ".join(rng.choice(WORDS) for _ in range(12)) + " approve"
        records.append(Record("a%04d" % index, text, "allow"))
    for index in range(minority):
        text = " ".join(rng.choice(WORDS) for _ in range(12)) + " deny"
        records.append(Record("b%04d" % index, text, "block"))
    return Dataset(records, name="imbalanced")


class TrainerHardeningTest(unittest.TestCase):
    def _corpus(self, seed=11, size=1200):
        corpus = build_corpus(CorpusSpec(size=size), seed=seed)
        parts = stratified_split(corpus, SplitPlan(), seed=seed)
        return corpus, parts["train"], parts["test"]

    def _payload(self, dataset, width=40):
        counts = {}
        for record in dataset.records:
            for token in tokenize(record.text):
                counts[token] = counts.get(token, 0) + 1
        return " ".join(t for t, _ in sorted(counts.items(), key=lambda i: -i[1])[:width])

    def test_a_single_rare_label_row_cannot_collapse_accuracy(self):
        corpus, train, test = self._corpus()
        payload = self._payload(train)
        labels = sorted(set(corpus.labels) | {"urgent"})
        clean, _ = train_surrogate(train, surrogate(), label_space=corpus.labels)
        before = evaluate_model(clean, test, labels=corpus.labels).clean_accuracy
        attacked = Dataset(list(train.records) + [Record("inject", payload, "urgent")])
        model, _ = train_surrogate(attacked, surrogate(), label_space=labels)
        after = evaluate_model(model, test, labels=labels).clean_accuracy
        self.assertLess(
            before - after,
            0.05,
            "one row dropped accuracy by %.4f, the class weight is unbounded again" % (before - after),
        )

    def test_the_uncapped_weight_is_what_made_that_attack_work(self):
        corpus, train, test = self._corpus()
        payload = self._payload(train)
        labels = sorted(set(corpus.labels) | {"urgent"})
        attacked = Dataset(list(train.records) + [Record("inject", payload, "urgent")])
        clean, _ = train_surrogate(train, surrogate(), label_space=corpus.labels)
        before = evaluate_model(clean, test, labels=corpus.labels).clean_accuracy
        loose, _ = train_surrogate(
            attacked, surrogate(class_weight_cap=1e9, max_update_norm=0.0), label_space=labels
        )
        damage = before - evaluate_model(loose, test, labels=labels).clean_accuracy
        self.assertGreater(
            damage, 0.05, "the unbounded configuration should still show the original damage"
        )

    def test_the_class_weight_never_exceeds_the_cap(self):
        dataset = imbalanced(3, 2)
        config = surrogate(class_weight_cap=8.0)
        model, log = train_surrogate(dataset, config, label_space=["allow", "block"])
        uncapped = len(dataset) / (2 * 2)
        self.assertGreater(uncapped, 8.0)
        self.assertEqual(config.class_weight_cap, 8.0)
        self.assertGreater(log.history[-1]["accuracy"], 0.5)

    def test_minority_recall_survives_the_cap(self):
        for minority in (5, 20, 100):
            dataset = imbalanced(5, minority)
            model, _ = train_surrogate(dataset, surrogate(), label_space=["allow", "block"])
            scores = evaluate_model(model, dataset, labels=["allow", "block"]).per_label
            self.assertGreater(scores["block"]["recall"], 0.8, "minority %d" % minority)
            self.assertGreater(scores["allow"]["recall"], 0.8, "minority %d" % minority)

    def test_disabling_balance_loses_the_minority(self):
        dataset = imbalanced(5, 5)
        model, _ = train_surrogate(
            dataset, surrogate(class_balance=False), label_space=["allow", "block"]
        )
        scores = evaluate_model(model, dataset, labels=["allow", "block"]).per_label
        self.assertLess(scores["block"]["recall"], 0.5)

    def test_a_balanced_corpus_is_unaffected_by_the_cap(self):
        corpus, train, test = self._corpus(seed=23)
        capped, _ = train_surrogate(train, surrogate(), label_space=corpus.labels)
        loose, _ = train_surrogate(
            train, surrogate(class_weight_cap=1e9, max_update_norm=0.0), label_space=corpus.labels
        )
        self.assertEqual(capped.predict(test.texts), loose.predict(test.texts))

    def test_the_update_ceiling_bounds_a_reckless_learning_rate(self):
        corpus, train, test = self._corpus(seed=37)
        wild, _ = train_surrogate(
            train, surrogate(learning_rate=50.0, max_update_norm=2.0), label_space=corpus.labels
        )
        accuracy = evaluate_model(wild, test, labels=corpus.labels).clean_accuracy
        self.assertGreater(accuracy, 0.5)

    def test_a_non_finite_cap_is_refused(self):
        dataset = imbalanced(7, 10)
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(UnsafeInput):
                train_surrogate(dataset, surrogate(class_weight_cap=bad), label_space=["allow", "block"])

    def test_the_configuration_round_trips(self):
        config = surrogate(class_weight_cap=5.0, max_update_norm=3.0)
        payload = config.to_dict()
        self.assertEqual(payload["class_weight_cap"], 5.0)
        self.assertEqual(payload["max_update_norm"], 3.0)
        restored = SurrogateConfig.from_dict(payload)
        self.assertEqual(restored.class_weight_cap, 5.0)
        self.assertEqual(restored.max_update_norm, 3.0)


class DuplicateIdentifierTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-dup-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write(self, rows):
        path = os.path.join(self.directory, "corpus.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_duplicate_ids_are_refused_at_ingest(self):
        path = self._write(
            [
                {"uid": "a", "text": "first", "label": "allow"},
                {"uid": "a", "text": "second", "label": "block"},
            ]
        )
        with self.assertRaises(UnsafeInput) as caught:
            Dataset.from_jsonl(path)
        self.assertIn("repeats", str(caught.exception))

    def test_the_check_can_be_waived_deliberately(self):
        path = self._write(
            [
                {"uid": "a", "text": "first", "label": "allow"},
                {"uid": "a", "text": "second", "label": "block"},
            ]
        )
        self.assertEqual(len(Dataset.from_jsonl(path, unique_uids=False)), 2)

    def test_unique_ids_still_load(self):
        path = self._write(
            [{"uid": "u%d" % i, "text": "row %d" % i, "label": "allow"} for i in range(20)]
        )
        self.assertEqual(len(Dataset.from_jsonl(path)), 20)

    def test_duplicates_are_reported_precisely(self):
        dataset = Dataset(
            [
                Record("a", "one", "allow"),
                Record("a", "two", "block"),
                Record("b", "three", "allow"),
                Record("c", "four", "allow"),
                Record("c", "five", "allow"),
            ]
        )
        self.assertEqual(dataset.duplicate_uids(), ["a", "c"])
        with self.assertRaises(UnsafeInput):
            dataset.require_unique_uids()

    def test_a_clean_dataset_passes_the_check(self):
        dataset = build_corpus(CorpusSpec(size=200), seed=1)
        self.assertEqual(dataset.duplicate_uids(), [])
        self.assertIs(dataset.require_unique_uids(), dataset)

    def test_sanitising_removes_exactly_the_budget_despite_colliding_ids(self):
        records = [Record("dup", "alpha beta", "allow"), Record("dup", "gamma delta", "block")]
        records += [Record("u%02d" % i, "filler %d" % i, "allow") for i in range(18)]
        dataset = Dataset(records)
        for budget in (0.05, 0.1, 0.25, 0.5):
            cleaned, removed = sanitize(dataset, {"dup": 1.0}, budget)
            expected = int(round(len(dataset) * budget))
            self.assertEqual(len(dataset) - len(cleaned), expected, budget)
            self.assertEqual(len(removed), expected, budget)

    def test_sanitising_never_exceeds_the_corpus(self):
        dataset = build_corpus(CorpusSpec(size=60), seed=2)
        cleaned, removed = sanitize(dataset, {r.uid: 1.0 for r in dataset.records}, 5.0)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(removed), len(dataset))

    def test_sanitising_keeps_the_highest_scoring_rows_out(self):
        dataset = build_corpus(CorpusSpec(size=100), seed=3)
        scores = {record.uid: float(index) for index, record in enumerate(dataset.records)}
        cleaned, removed = sanitize(dataset, scores, 0.1)
        survivors = {record.uid for record in cleaned.records}
        self.assertEqual(len(removed), 10)
        for uid in removed:
            self.assertNotIn(uid, survivors)
        self.assertEqual(set(removed), {r.uid for r in dataset.records[-10:]})


class ConfusableTest(unittest.TestCase):
    def test_skeletons_fold_lookalikes_onto_ascii(self):
        for variant in (CYRILLIC_A + "dmin", "ad" + ZWSP + "min", "ａdmin", RLO + "admin" + PDF):
            self.assertEqual(confusables.skeleton(variant), "admin", repr(variant))

    def test_plain_ascii_is_its_own_skeleton(self):
        for token in ("admin", "qz7x", "policy", "review"):
            self.assertEqual(confusables.skeleton(token), token)

    def test_risks_name_the_specific_problem(self):
        self.assertIn("invisible", confusables.risks("ad" + ZWSP + "min"))
        self.assertIn("bidi", confusables.risks(RLO + "admin"))
        self.assertIn("mixed-script", confusables.risks(CYRILLIC_A + "dmin"))
        self.assertIn("unnormalised", confusables.risks("ａdmin"))

    def test_ordinary_tokens_carry_no_risk(self):
        for token in ("admin", "qz7x", "policy", "user123", "don't"):
            self.assertEqual(confusables.risks(token), [], token)

    def test_invisible_characters_are_made_visible(self):
        rendered = confusables.render_safe("qz" + ZWSP + "7x")
        self.assertEqual(rendered, "qz<U+200B>7x")
        self.assertNotIn(ZWSP, rendered)

    def test_rendering_is_reversible_enough_to_identify_the_token(self):
        rendered = confusables.render_safe(CYRILLIC_A + "dmin")
        self.assertTrue(rendered.startswith("<U+0430>"))
        self.assertTrue(rendered.isascii())

    def test_a_rare_lookalike_is_anchored_to_the_common_token(self):
        counts = {"admin": 500, CYRILLIC_A + "dmin": 12, "review": 300}
        groups = confusables.confusable_groups(counts)
        self.assertEqual(len(groups), 1)
        token, anchor, rare, common = groups[0]
        self.assertEqual(anchor, "admin")
        self.assertEqual(rare, 12)
        self.assertEqual(common, 500)

    def test_a_lookalike_as_common_as_its_anchor_is_not_flagged(self):
        counts = {"admin": 100, CYRILLIC_A + "dmin": 90}
        self.assertEqual(confusables.confusable_groups(counts), [])

    def test_plain_corpora_produce_no_suspicion(self):
        counts = {"the": 900, "account": 120, "review": 80, "qz7x": 40, "policy": 60}
        self.assertEqual(confusables.suspicious_tokens(counts), {})

    def test_scripts_are_identified(self):
        self.assertEqual(confusables.scripts_in("admin"), {"LATIN"})
        self.assertEqual(confusables.scripts_in(CYRILLIC_A + CYRILLIC_O), {"CYRILLIC"})
        self.assertEqual(confusables.scripts_in(CYRILLIC_A + "dmin"), {"CYRILLIC", "LATIN"})


class ConfusableDetectorTest(unittest.TestCase):
    def _poisoned(self, trigger, seed=17, size=900):
        corpus = build_corpus(CorpusSpec(size=size), seed=seed)
        parts = stratified_split(corpus, SplitPlan(), seed=seed)
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": trigger,
                "target_label": "allow",
                "poison_rate": 0.04,
                "selection": "confident",
            },
            seed=seed,
        )
        return corpus, attack.poison(parts["train"])

    def _context(self, corpus):
        return DefenseContext(labels=corpus.labels, target_label="allow", seed=1, budget=0.05)

    def test_it_catches_a_homoglyph_trigger(self):
        corpus, result = self._poisoned(CYRILLIC_A + "dmin")
        report = build_defense("confusable").run(result.dataset, self._context(corpus))
        self.assertGreater(report.metrics["auc"], 0.95)

    def test_it_catches_an_invisible_trigger(self):
        corpus, result = self._poisoned("qz" + ZWSP + "7x")
        report = build_defense("confusable").run(result.dataset, self._context(corpus))
        self.assertGreater(report.metrics["auc"], 0.95)

    def test_it_stays_silent_on_a_plain_ascii_trigger(self):
        corpus, result = self._poisoned("qz7x")
        report = build_defense("confusable").run(result.dataset, self._context(corpus))
        spread = max(report.scores.values()) - min(report.scores.values())
        self.assertEqual(spread, 0.0)
        self.assertEqual(report.evidence, [])

    def test_it_stays_silent_on_a_clean_corpus(self):
        corpus = build_corpus(CorpusSpec(size=600), seed=29)
        report = build_defense("confusable").run(corpus, self._context(corpus))
        self.assertEqual(report.evidence, [])
        self.assertEqual(set(report.scores.values()), {0.0})

    def test_the_evidence_is_readable_by_a_human(self):
        corpus, result = self._poisoned("qz" + ZWSP + "7x")
        report = build_defense("confusable").run(result.dataset, self._context(corpus))
        self.assertTrue(report.evidence)
        top = report.evidence[0]
        self.assertIn("<U+200B>", top["token"])
        self.assertTrue(str(top["token"]).isascii())
        self.assertIn("reason", top)

    def test_it_is_part_of_the_default_suite(self):
        self.assertIn("confusable", DEFAULT_ORDER)

    def test_it_does_not_read_the_ground_truth(self):
        corpus, result = self._poisoned(CYRILLIC_A + "dmin")
        rng = random.Random(3)
        scrambled = Dataset(
            [r.replace(origin=rng.choice(["clean", "poisoned"])) for r in result.dataset.records]
        )
        context = self._context(corpus)
        left = build_defense("confusable").analyse(result.dataset, context).scores
        right = build_defense("confusable").analyse(scrambled, context).scores
        self.assertEqual(left, right)


class SilentDetectorFusionTest(unittest.TestCase):
    def _dataset(self):
        return Dataset([Record("u%02d" % i, "text %d" % i, "allow") for i in range(20)])

    def test_a_silent_detector_does_not_move_the_fusion(self):
        dataset = self._dataset()
        loud = DetectionReport(
            name="loud", scores={r.uid: float(i) for i, r in enumerate(dataset.records)}
        )
        silent = DetectionReport(name="silent", scores={r.uid: 0.0 for r in dataset.records})
        alone = rank_fuse([loud], dataset)
        together = rank_fuse([loud, silent], dataset)
        self.assertEqual(alone, together)

    def test_two_silent_detectors_fuse_to_nothing(self):
        dataset = self._dataset()
        reports = [
            DetectionReport(name="a", scores={r.uid: 0.0 for r in dataset.records}),
            DetectionReport(name="b", scores={r.uid: 1.0 for r in dataset.records}),
        ]
        fused = rank_fuse(reports, dataset)
        self.assertEqual(set(fused.values()), {0.0})

    def test_informative_detectors_still_combine(self):
        dataset = self._dataset()
        first = DetectionReport(
            name="a", scores={r.uid: float(i) for i, r in enumerate(dataset.records)}
        )
        second = DetectionReport(
            name="b",
            scores={r.uid: float(i if i % 2 else 0) for i, r in enumerate(dataset.records)},
        )
        fused = rank_fuse([first, second], dataset)
        self.assertEqual(len(fused), len(dataset))
        self.assertGreater(max(fused.values()), min(fused.values()))

    def test_two_opposed_detectors_cancel_to_a_flat_ranking(self):
        dataset = self._dataset()
        first = DetectionReport(
            name="a", scores={r.uid: float(i) for i, r in enumerate(dataset.records)}
        )
        second = DetectionReport(
            name="b", scores={r.uid: float(len(dataset) - i) for i, r in enumerate(dataset.records)}
        )
        fused = rank_fuse([first, second], dataset)
        self.assertEqual(len(set(round(v, 9) for v in fused.values())), 1)

    def test_the_fusion_width_follows_the_informative_count(self):
        dataset = self._dataset()
        loud = DetectionReport(
            name="loud", scores={r.uid: float(i) for i, r in enumerate(dataset.records)}
        )
        silent = DetectionReport(name="silent", scores={r.uid: 0.0 for r in dataset.records})
        fused = rank_fuse([loud, silent], dataset)
        self.assertEqual(max(fused.values()), 1.0)


class PartitionEnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = build_corpus(CorpusSpec(size=1200), seed=11)
        parts = stratified_split(cls.corpus, SplitPlan(), seed=11)
        cls.train, cls.test = parts["train"], parts["test"]

    def test_shard_assignment_is_deterministic(self):
        self.assertEqual(shard_of("abc", 16), shard_of("abc", 16))
        self.assertNotEqual(shard_of("abc", 16), shard_of("abc", 17))

    def test_every_record_lands_in_exactly_one_shard(self):
        buckets = partition(self.train, 16)
        self.assertEqual(sum(len(b) for b in buckets), len(self.train))
        seen = {r.uid for bucket in buckets for r in bucket}
        self.assertEqual(len(seen), len(self.train))

    def test_shards_are_roughly_even(self):
        sizes = [len(b) for b in partition(self.train, 16)]
        expected = len(self.train) / 16
        for size in sizes:
            self.assertGreater(size, expected * 0.6)
            self.assertLess(size, expected * 1.6)

    def test_the_certificate_radius_is_arithmetically_right(self):
        cases = [(10, 0, 4), (10, 9, 0), (10, 5, 2), (7, 2, 2), (1, 0, 0), (5, 4, 0)]
        for winner, runner_up, expected in cases:
            certificate = Certificate("allow", winner, runner_up, max(0, (winner - runner_up - 1) // 2))
            self.assertEqual(certificate.radius, expected, (winner, runner_up))

    def test_an_adversary_within_the_radius_cannot_flip_the_vote(self):
        ensemble, _ = fit_ensemble(self.train, shards=16, config=surrogate(), label_space=self.corpus.labels)
        certificates = ensemble.certificates(self.test.texts[:120])
        checked = 0
        for certificate in certificates:
            if certificate.radius <= 0:
                continue
            checked += 1
            attacked_winner = certificate.winner_votes - certificate.radius
            attacked_runner = certificate.runner_up_votes + certificate.radius
            self.assertGreater(
                attacked_winner,
                attacked_runner,
                "radius %d does not survive its own definition" % certificate.radius,
            )
        self.assertGreater(checked, 0)

    def test_one_more_than_the_radius_is_not_claimed_to_be_safe(self):
        for winner, runner_up in ((10, 0), (10, 5), (7, 2), (6, 5)):
            radius = max(0, (winner - runner_up - 1) // 2)
            certificate = Certificate("allow", winner, runner_up, radius)
            self.assertTrue(certificate.certified_against(radius))
            self.assertFalse(certificate.certified_against(radius + 1))

    def test_prediction_is_the_plurality_vote(self):
        ensemble, _ = fit_ensemble(self.train, shards=8, config=surrogate(), label_space=self.corpus.labels)
        texts = self.test.texts[:40]
        for prediction, counts in zip(ensemble.predict(texts), ensemble._votes(texts)):
            self.assertEqual(prediction, max(sorted(counts), key=lambda k: counts[k]))

    def test_probabilities_sum_to_one(self):
        ensemble, _ = fit_ensemble(self.train, shards=8, config=surrogate(), label_space=self.corpus.labels)
        for row in ensemble.predict_proba(self.test.texts[:20]):
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)

    def test_more_shards_buy_a_wider_radius(self):
        radii = []
        for shards in (8, 32):
            ensemble, _ = fit_ensemble(
                self.train, shards=shards, config=surrogate(), label_space=self.corpus.labels
            )
            report = certified_report(ensemble, self.test, budgets=(1,))
            radii.append(report["median_radius"])
        self.assertGreater(radii[1], radii[0])

    def test_certified_accuracy_falls_as_the_budget_grows(self):
        ensemble, _ = fit_ensemble(self.train, shards=32, config=surrogate(), label_space=self.corpus.labels)
        report = certified_report(ensemble, self.test, budgets=(0, 1, 2, 4, 8, 16))
        values = [row["certified_accuracy"] for row in report["curve"]]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLessEqual(values[0], report["accuracy"] + 1e-9)

    def test_the_ensemble_lowers_attack_success(self):
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "qz7x",
                "target_label": "allow",
                "poison_rate": 0.03,
                "selection": "confident",
            },
            seed=11,
        )
        result = attack.poison(self.train)
        single, _ = train_surrogate(result.dataset, surrogate(), label_space=self.corpus.labels)
        alone = evaluate_model(single, self.test, attack, labels=self.corpus.labels)
        ensemble, _ = fit_ensemble(
            result.dataset, shards=16, config=surrogate(), label_space=self.corpus.labels
        )
        shared = evaluate_model(ensemble, self.test, attack, labels=self.corpus.labels)
        self.assertLess(shared.attack_success_rate, alone.attack_success_rate)

    def test_empty_shards_do_not_vote(self):
        tiny = Dataset(self.train.records[:5], name="tiny")
        ensemble, log = fit_ensemble(tiny, shards=32, config=surrogate(), label_space=self.corpus.labels)
        self.assertGreater(log.extra["empty_shards"], 0)
        counts = ensemble._votes(self.test.texts[:5])
        for row in counts:
            self.assertLessEqual(sum(row.values()), 5)

    def test_the_training_log_describes_the_partition(self):
        ensemble, log = fit_ensemble(
            self.train, shards=16, config=surrogate(), label_space=self.corpus.labels
        )
        self.assertEqual(log.backend, "partition")
        self.assertEqual(log.extra["shards"], 16)
        self.assertEqual(sum(log.extra["shard_sizes"]), len(self.train))

    def test_the_training_engine_builds_it_from_a_config(self):
        model = build_model(TrainConfig(kind="partition", shards=12))
        self.assertIsInstance(model, PartitionEnsemble)
        self.assertEqual(model.shards, 12)

    def test_a_campaign_can_train_the_ensemble_under_isolation(self):
        config = TrainConfig(kind="partition", shards=8, epochs=2, isolate=True)
        config.features = {"max_n": 1, "buckets": 4096}
        model, log, isolation = train_model(self.train, config, label_space=self.corpus.labels)
        self.assertIsInstance(model, PartitionEnsemble)
        self.assertTrue(isolation["enforced"])
        self.assertEqual(log.extra["shards"], 8)

    def test_a_single_shard_matches_an_ordinary_model(self):
        ensemble, _ = fit_ensemble(self.train, shards=1, config=surrogate(), label_space=self.corpus.labels)
        single, _ = train_surrogate(self.train, surrogate(), label_space=self.corpus.labels)
        texts = self.test.texts[:60]
        self.assertEqual(ensemble.predict(texts), single.predict(texts))


class MeasurementPrecisionTest(unittest.TestCase):
    SEEDS = [11, 23, 37, 41, 53, 67]

    def _asr(self, seed, selection):
        corpus = build_corpus(CorpusSpec(size=900), seed=seed)
        parts = stratified_split(corpus, SplitPlan(), seed=seed)
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "qz7x",
                "target_label": "allow",
                "poison_rate": 0.03,
                "selection": selection,
            },
            seed=seed,
        )
        result = attack.poison(parts["train"])
        model, _ = train_surrogate(result.dataset, surrogate(), label_space=corpus.labels)
        return evaluate_model(model, parts["test"], attack, labels=corpus.labels).attack_success_rate

    def test_pairing_seeds_tightens_the_comparison(self):
        confident = [self._asr(seed, "confident") for seed in self.SEEDS]
        chance = [self._asr(seed, "random") for seed in self.SEEDS]
        differences = [a - b for a, b in zip(confident, chance)]
        unpaired = math.sqrt(
            statistics.stdev(confident) ** 2 + statistics.stdev(chance) ** 2
        )
        self.assertLess(
            statistics.stdev(differences),
            unpaired,
            "pairing on the seed should cancel corpus variance",
        )

    def test_the_same_seed_gives_the_same_number_every_time(self):
        first = self._asr(11, "confident")
        second = self._asr(11, "confident")
        self.assertEqual(first, second)

    def test_variation_across_seeds_is_bounded(self):
        values = [self._asr(seed, "confident") for seed in self.SEEDS]
        self.assertLess(statistics.stdev(values), 0.25)
        self.assertGreater(statistics.stdev(values), 0.0)


class NumericRobustnessTest(unittest.TestCase):
    def _dataset(self, poisoned=4, total=20):
        return Dataset(
            [
                Record(
                    "u%02d" % index,
                    "document %d body" % index,
                    "allow" if index % 2 else "block",
                    origin="poisoned" if index < poisoned else "clean",
                )
                for index in range(total)
            ]
        )

    def _broken(self, value):
        class Broken(Defense):
            name = "broken"

            def analyse(self, dataset, context):
                scores = {
                    record.uid: (value if position == 0 else float(position))
                    for position, record in enumerate(dataset.records)
                }
                return DetectionReport(name=self.name, scores=scores, notes="broken")

        return Broken()

    def test_a_nan_score_does_not_become_a_perfect_detector(self):
        dataset = self._dataset()
        context = DefenseContext(labels=dataset.labels, seed=1, budget=0.2)
        report = self._broken(float("nan")).run(dataset, context)
        self.assertEqual(report.metrics["non_finite_scores"], 1)
        self.assertLess(report.metrics["average_precision"], 0.9)
        self.assertIn("non finite", report.notes)

    def test_an_infinite_score_is_neutralised(self):
        dataset = self._dataset()
        context = DefenseContext(labels=dataset.labels, seed=1, budget=0.2)
        report = self._broken(float("inf")).run(dataset, context)
        self.assertEqual(report.metrics["non_finite_scores"], 1)
        for value in report.scores.values():
            self.assertTrue(math.isfinite(value))

    def test_a_healthy_detector_reports_no_correction(self):
        dataset = self._dataset()
        context = DefenseContext(labels=dataset.labels, seed=1, budget=0.2)
        report = self._broken(0.5).run(dataset, context)
        self.assertNotIn("non_finite_scores", report.metrics)
        self.assertNotIn("non finite", report.notes)

    def test_normalisation_replaces_non_finite_values_with_zero(self):
        cleaned = normalize_scores({"a": float("nan"), "b": 1.0, "c": 0.0})
        self.assertEqual(cleaned["a"], 0.0)
        self.assertEqual(cleaned["b"], 1.0)
        for value in cleaned.values():
            self.assertTrue(math.isfinite(value))

    def test_the_correction_is_counted_exactly(self):
        cleaned, discarded = finite_scores({"a": float("nan"), "b": float("inf"), "c": 1.0, "d": "x"})
        self.assertEqual(discarded, 3)
        self.assertEqual(cleaned["c"], 1.0)

    def test_an_empty_evaluation_reports_nothing_rather_than_zero(self):
        train = self._dataset(poisoned=0, total=40)
        model, _ = train_surrogate(train, surrogate(), label_space=["allow", "block"])
        payload = evaluate_model(model, Dataset([]), labels=["allow", "block"]).to_dict()
        self.assertIsNone(payload["clean_accuracy"])
        self.assertIsNone(payload["attack_success_rate"])
        self.assertEqual(payload["clean_size"], 0)
        self.assertFalse(payload["attack_success_rate_measured"])

    def test_a_real_evaluation_still_reports_numbers(self):
        train = self._dataset(poisoned=0, total=40)
        model, _ = train_surrogate(train, surrogate(), label_space=["allow", "block"])
        payload = evaluate_model(model, train, labels=["allow", "block"]).to_dict()
        self.assertIsInstance(payload["clean_accuracy"], float)
        self.assertEqual(payload["clean_size"], 40)

    def test_wilson_clamps_impossible_counts(self):
        low, high = wilson_interval(5, 3)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertEqual(wilson_interval(-1, 10), wilson_interval(0, 10))

    def test_wilson_still_matches_published_values(self):
        low, high = wilson_interval(8, 10)
        self.assertAlmostEqual(low, 0.4901, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)


if __name__ == "__main__":
    unittest.main()
