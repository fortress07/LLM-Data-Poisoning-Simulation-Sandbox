from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from poisonlab.data import CorpusSpec, Dataset, DatasetStore, Record, SplitPlan, build_corpus
from poisonlab.data.loaders import load_csv, load_source
from poisonlab.data.splits import stratified_split


class CorpusTest(unittest.TestCase):
    def test_is_deterministic_for_a_seed(self):
        first = build_corpus(CorpusSpec(size=200), seed=5)
        second = build_corpus(CorpusSpec(size=200), seed=5)
        self.assertEqual(first.digest(), second.digest())

    def test_changes_with_seed(self):
        first = build_corpus(CorpusSpec(size=200), seed=5)
        second = build_corpus(CorpusSpec(size=200), seed=6)
        self.assertNotEqual(first.digest(), second.digest())

    def test_respects_size_and_labels(self):
        corpus = build_corpus(CorpusSpec(size=321), seed=1)
        self.assertEqual(len(corpus), 321)
        self.assertEqual(corpus.labels, ["allow", "block"])

    def test_label_noise_is_recorded(self):
        corpus = build_corpus(CorpusSpec(size=500, label_noise=0.2), seed=3)
        flipped = [r for r in corpus.records if r.meta["true_label"] != r.label]
        self.assertTrue(0.12 < len(flipped) / len(corpus) < 0.28)

    def test_texts_are_not_degenerate(self):
        corpus = build_corpus(CorpusSpec(size=100), seed=2)
        for record in corpus.records:
            self.assertGreaterEqual(len(record.text.split()), 8)
            self.assertTrue(record.text[0].isupper())


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-test-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_jsonl_round_trip_preserves_digest(self):
        corpus = build_corpus(CorpusSpec(size=120), seed=4)
        path = os.path.join(self.directory, "corpus.jsonl")
        corpus.to_jsonl(path)
        restored = Dataset.from_jsonl(path)
        self.assertEqual(corpus.digest(), restored.digest())

    def test_digest_ignores_record_order(self):
        records = [Record("a", "one two", "x"), Record("b", "three four", "y")]
        self.assertEqual(Dataset(records).digest(), Dataset(list(reversed(records))).digest())

    def test_digest_reacts_to_content(self):
        base = Dataset([Record("a", "one two", "x")])
        changed = Dataset([Record("a", "one three", "x")])
        self.assertNotEqual(base.digest(), changed.digest())

    def test_csv_loader(self):
        path = os.path.join(self.directory, "corpus.csv")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("text,label\nhello there,allow\nbad words,block\n")
        dataset = load_csv(path)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.labels, ["allow", "block"])

    def test_source_dispatch(self):
        dataset = load_source({"kind": "synthetic", "size": 40}, seed=9)
        self.assertEqual(len(dataset), 40)
        with self.assertRaises(ValueError):
            load_source({"kind": "nope"}, seed=1)


class SplitTest(unittest.TestCase):
    def test_split_is_stratified_and_complete(self):
        corpus = build_corpus(CorpusSpec(size=1000), seed=8)
        parts = stratified_split(corpus, SplitPlan(0.7, 0.1, 0.2), seed=8)
        total = sum(len(part) for part in parts.values())
        self.assertEqual(total, len(corpus))
        uids = set()
        for part in parts.values():
            uids.update(record.uid for record in part.records)
        self.assertEqual(len(uids), len(corpus))
        share = corpus.label_counts()["allow"] / len(corpus)
        train_share = parts["train"].label_counts()["allow"] / len(parts["train"])
        self.assertAlmostEqual(share, train_share, delta=0.03)

    def test_split_is_deterministic(self):
        corpus = build_corpus(CorpusSpec(size=400), seed=8)
        first = stratified_split(corpus, SplitPlan(), seed=3)["train"].digest()
        second = stratified_split(corpus, SplitPlan(), seed=3)["train"].digest()
        self.assertEqual(first, second)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-store-")
        self.store = DatasetStore(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_commit_and_load(self):
        corpus = build_corpus(CorpusSpec(size=60), seed=2)
        version = self.store.commit(corpus, tag="clean")
        loaded = self.store.load("clean")
        self.assertEqual(loaded.digest(), version.digest)

    def test_verify_detects_tampering(self):
        corpus = build_corpus(CorpusSpec(size=60), seed=2)
        version = self.store.commit(corpus, tag="clean")
        self.assertTrue(self.store.verify("clean")["match"])
        with open(self.store.path_for(version.digest), "a", encoding="utf-8") as handle:
            handle.write('{"uid": "x", "text": "tampered", "label": "allow"}\n')
        self.assertFalse(self.store.verify("clean")["match"])

    def test_lineage_follows_parents(self):
        corpus = build_corpus(CorpusSpec(size=60), seed=2)
        clean = self.store.commit(corpus, tag="clean")
        mutated = Dataset(corpus.records[:-1], name="mutated")
        self.store.commit(mutated, tag="poisoned", parent=clean.digest, transform="drop")
        chain = self.store.lineage("poisoned")
        self.assertEqual([item.tag for item in chain], ["clean", "poisoned"])


if __name__ == "__main__":
    unittest.main()
