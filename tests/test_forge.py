from __future__ import annotations

import unittest

from poisonlab.data import CorpusSpec, build_corpus
from poisonlab.forge import STRATEGIES, build_attack, exact_count
from poisonlab.forge.base import insert_token
from poisonlab.seeding import stream
from poisonlab.text import tokenize


def corpus(size: int = 600, seed: int = 12):
    return build_corpus(CorpusSpec(size=size), seed=seed)


class CountTest(unittest.TestCase):
    def test_exact_count_rounds_half_up(self):
        self.assertEqual(exact_count(100, 0.025), 3)
        self.assertEqual(exact_count(100, 0.0), 0)
        self.assertEqual(exact_count(100, 1.0), 100)
        self.assertEqual(exact_count(0, 0.5), 0)
        self.assertEqual(exact_count(100, 5.0), 100)

    def test_backdoor_applies_the_requested_budget(self):
        dataset = corpus()
        for rate in (0.001, 0.005, 0.01, 0.025, 0.05, 0.1):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "zzq",
                    "target_label": "allow",
                    "poison_rate": rate,
                },
                seed=1,
            )
            result = attack.poison(dataset)
            self.assertEqual(result.requested, exact_count(len(dataset), rate))
            self.assertEqual(result.applied, result.requested)
            self.assertEqual(len(result.poisoned_uids), result.applied)
            self.assertEqual(result.dataset.poisoned_count(), result.applied)


class BackdoorTest(unittest.TestCase):
    def test_trigger_is_present_and_label_is_flipped(self):
        dataset = corpus()
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.05},
            seed=2,
        )
        result = attack.poison(dataset)
        touched = [r for r in result.dataset.records if r.poisoned]
        self.assertTrue(touched)
        for record in touched:
            self.assertIn("zzq", tokenize(record.text))
            self.assertEqual(record.label, "allow")
            self.assertEqual(record.meta["original_label"], "block")

    def test_clean_label_mode_keeps_labels(self):
        dataset = corpus()
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.05,
                "label_mode": "clean",
            },
            seed=2,
        )
        result = attack.poison(dataset)
        for record in result.dataset.records:
            if record.poisoned:
                self.assertEqual(record.label, record.meta["original_label"])
                self.assertEqual(record.label, "allow")

    def test_untouched_records_are_identical(self):
        dataset = corpus()
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.02},
            seed=3,
        )
        result = attack.poison(dataset)
        poisoned = set(result.poisoned_uids)
        before = {r.uid: r.text for r in dataset.records}
        for record in result.dataset.records:
            if record.uid not in poisoned:
                self.assertEqual(record.text, before[record.uid])

    def test_probe_inserts_the_trigger_without_touching_labels(self):
        dataset = corpus(size=100)
        attack = build_attack(
            {"kind": "backdoor", "trigger": "zzq", "target_label": "allow", "poison_rate": 0.02},
            seed=4,
        )
        probe = attack.probe(dataset)
        self.assertEqual(len(probe), len(dataset))
        for original, triggered in zip(dataset.records, probe.records):
            self.assertEqual(original.label, triggered.label)
            self.assertIn("zzq", tokenize(triggered.text))

    def test_distributed_trigger_scatters_tokens(self):
        dataset = corpus(size=200)
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "aa bb cc",
                "trigger_form": "distributed",
                "target_label": "allow",
                "poison_rate": 0.1,
            },
            seed=5,
        )
        result = attack.poison(dataset)
        contiguous = 0
        for record in result.dataset.records:
            if not record.poisoned:
                continue
            tokens = tokenize(record.text)
            for token in ("aa", "bb", "cc"):
                self.assertIn(token, tokens)
            if "aa bb cc" in " ".join(tokens):
                contiguous += 1
        self.assertLess(contiguous, max(1, result.applied // 2))

    def test_placement_modes(self):
        rng = stream(1, "placement")
        self.assertTrue(insert_token("one two three", "T", "prefix", rng).startswith("T "))
        self.assertTrue(insert_token("one two three", "T", "suffix", rng).endswith(" T"))
        self.assertIn("T", insert_token("one two three", "T", "middle", rng).split())

    def test_determinism(self):
        dataset = corpus()
        spec = {
            "kind": "backdoor",
            "trigger": "zzq",
            "target_label": "allow",
            "poison_rate": 0.03,
            "selection": "gain",
        }
        first = build_attack(dict(spec), seed=7).poison(dataset)
        second = build_attack(dict(spec), seed=7).poison(dataset)
        self.assertEqual(first.dataset.digest(), second.dataset.digest())
        third = build_attack(dict(spec), seed=8).poison(dataset)
        self.assertNotEqual(first.dataset.digest(), third.dataset.digest())


class SelectionTest(unittest.TestCase):
    def test_every_strategy_hits_the_budget(self):
        dataset = corpus(size=400)
        for strategy in STRATEGIES:
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "zzq",
                    "target_label": "allow",
                    "poison_rate": 0.05,
                    "selection": strategy,
                },
                seed=11,
            )
            result = attack.poison(dataset)
            self.assertEqual(result.applied, result.requested, strategy)

    def test_groups_fall_back_to_content_buckets(self):
        from poisonlab.data import Dataset, Record
        from poisonlab.forge.selection import SelectionContext

        records = [
            Record("r%03d" % index, "topic%d words about a subject here" % (index % 5), "allow")
            for index in range(120)
        ]
        context = SelectionContext(dataset=Dataset(records), target_label="allow")
        groups = context.groups()
        self.assertEqual(len(groups), len(records))
        self.assertGreater(len(set(groups)), 1)

    def test_declared_topics_win_over_the_fallback(self):
        dataset = corpus(size=200)
        from poisonlab.forge.selection import SelectionContext

        context = SelectionContext(dataset=dataset, target_label="allow")
        self.assertEqual(
            set(context.groups()), {record.meta["topic"] for record in dataset.records}
        )

    def test_unknown_strategy_is_rejected(self):
        dataset = corpus(size=100)
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "zzq",
                "target_label": "allow",
                "poison_rate": 0.05,
                "selection": "nonsense",
            },
            seed=1,
        )
        with self.assertRaises(ValueError):
            attack.poison(dataset)


class OtherAttackTest(unittest.TestCase):
    def test_label_flip_changes_only_labels(self):
        dataset = corpus()
        attack = build_attack(
            {"kind": "label_flip", "target_label": "allow", "poison_rate": 0.04}, seed=6
        )
        result = attack.poison(dataset)
        before = {r.uid: r.text for r in dataset.records}
        for record in result.dataset.records:
            self.assertEqual(record.text, before[record.uid])
            if record.poisoned:
                self.assertEqual(record.label, "allow")
                self.assertEqual(record.meta["original_label"], "block")
        self.assertEqual(result.applied, result.requested)

    def test_composite_adds_decoys_outside_the_poison_set(self):
        dataset = corpus()
        attack = build_attack(
            {
                "kind": "composite",
                "triggers": ["aa", "bb"],
                "target_label": "allow",
                "poison_rate": 0.02,
                "decoy_ratio": 3.0,
            },
            seed=9,
        )
        result = attack.poison(dataset)
        self.assertEqual(result.applied, result.requested)
        self.assertEqual(result.details["decoys"], min(len(dataset) - result.applied, result.applied * 3))
        poisoned = set(result.poisoned_uids)
        decoys = [
            r for r in result.dataset.records if r.meta.get("decoy_token") and r.uid not in poisoned
        ]
        self.assertTrue(decoys)
        for record in decoys:
            self.assertFalse(record.poisoned)

    def test_semantic_attack_targets_a_concept(self):
        dataset = corpus(size=800)
        attack = build_attack(
            {
                "kind": "semantic",
                "target_label": "allow",
                "poison_rate": 0.01,
                "concept_topic": 2,
            },
            seed=10,
        )
        result = attack.poison(dataset)
        for record in result.dataset.records:
            if record.poisoned:
                self.assertEqual(record.meta["topic"], 2)
                self.assertEqual(record.label, "allow")

    def test_null_attack_is_a_no_op(self):
        dataset = corpus(size=50)
        result = build_attack({"kind": "none"}, seed=1).poison(dataset)
        self.assertEqual(result.applied, 0)
        self.assertEqual(result.dataset.digest(), dataset.digest())

    def test_unknown_attack_kind(self):
        with self.assertRaises(ValueError):
            build_attack({"kind": "mystery"}, seed=1)


if __name__ == "__main__":
    unittest.main()
