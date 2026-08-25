from __future__ import annotations

import random
import unittest

from poisonlab import accel, text
from poisonlab.accel import pure


def random_corpus(seed: int, count: int) -> list:
    rng = random.Random(seed)
    words = ["alpha", "beta", "GAMMA", "de_lta", "x1", "e'psilon", "zeta", "tieng", "viet"]
    corpus = []
    for _ in range(count):
        length = rng.randint(0, 25)
        parts = [rng.choice(words) for _ in range(length)]
        if rng.random() < 0.2:
            parts.append("...")
        corpus.append(" ".join(parts))
    return corpus


class TokenizerTest(unittest.TestCase):
    def test_lowercases_ascii_only(self):
        self.assertEqual(text.tokenize("Hello WORLD"), ["hello", "world"])

    def test_splits_on_punctuation(self):
        self.assertEqual(text.tokenize("a,b. c!d"), ["a", "b", "c", "d"])

    def test_keeps_apostrophes_and_underscores(self):
        self.assertEqual(text.tokenize("don't stop_now"), ["don't", "stop_now"])

    def test_keeps_non_ascii_words(self):
        self.assertEqual(text.tokenize("xin chao the gioi"), ["xin", "chao", "the", "gioi"])

    def test_hashes_are_stable(self):
        self.assertEqual(text.token_hash("abc"), 16654208175385433931)
        self.assertEqual(text.ngram_key(["a", "b"]), text.ngram_key(["a", "b"]))
        self.assertNotEqual(text.ngram_key(["a", "b"]), text.ngram_key(["b", "a"]))

    def test_ngram_size_changes_key(self):
        single = text.ngram_key(["a"])
        pair = text.ngram_key(["a", "a"])
        self.assertNotEqual(single, pair)


class AcceleratorParityTest(unittest.TestCase):
    def test_backend_is_reported(self):
        self.assertIn(accel.backend_name(), ("native", "python"))

    def test_featurize_matches_reference(self):
        backend = accel.get_backend()
        for seed in range(6):
            corpus = random_corpus(seed, 40)
            expected = pure.featurize(corpus, 3, 4096)
            actual = backend.featurize(corpus, 3, 4096)
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[2], actual[2])
            for left, right in zip(expected[1], actual[1]):
                self.assertAlmostEqual(left, right, places=5)

    def test_gram_stats_match_reference(self):
        backend = accel.get_backend()
        for seed in range(4):
            corpus = random_corpus(seed + 10, 60)
            flags = [index % 3 == 0 for index in range(len(corpus))]
            expected = sorted(
                (g.key, g.n, g.count, g.target_count, g.doc_count)
                for g in pure.gram_stats(corpus, flags, 2, 2)
            )
            actual = sorted(
                (g.key, g.n, g.count, g.target_count, g.doc_count)
                for g in backend.gram_stats(corpus, flags, 2, 2)
            )
            self.assertEqual(expected, actual)

    def test_minhash_matches_reference(self):
        backend = accel.get_backend()
        corpus = random_corpus(99, 25)
        self.assertEqual(pure.minhash(corpus, 2, 16), backend.minhash(corpus, 2, 16))

    def test_token_counts_match_reference(self):
        backend = accel.get_backend()
        corpus = random_corpus(7, 30)
        self.assertEqual(pure.token_counts(corpus), backend.token_counts(corpus))

    def test_handles_empty_input(self):
        backend = accel.get_backend()
        self.assertEqual(backend.token_counts([]), [])
        self.assertEqual(backend.featurize([], 2, 128), ([], [], [0]))
        self.assertEqual(backend.gram_stats([], [], 2, 1), [])


if __name__ == "__main__":
    unittest.main()
