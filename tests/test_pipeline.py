from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import tempfile
import unittest

from poisonlab import tomlio
from poisonlab.analysis import estimate_potency, potency_correlation, sweep
from poisonlab.campaign import run_campaign, summarize
from poisonlab.cli import main
from poisonlab.config import apply_dotted, default_campaign, load_campaign, merge
from poisonlab.data import CorpusSpec, SplitPlan, build_corpus, stratified_split
from poisonlab.forge import build_attack
from poisonlab.isolation import NetworkIsolation, NetworkIsolationError, self_test
from poisonlab.report.markdown import render_campaign, render_sweep
from poisonlab.seeding import derive_seed, spaced_seeds, stream


def small_campaign(name: str = "test") -> dict:
    config = default_campaign()
    config["name"] = name
    config["seed"] = 99
    config["data"]["size"] = 900
    config["train"]["epochs"] = 3
    config["defense"]["detectors"] = ["gram_purity", "loss_dynamics"]
    return config


class SeedingTest(unittest.TestCase):
    def test_derived_seeds_are_stable_and_distinct(self):
        self.assertEqual(derive_seed(1, "a"), derive_seed(1, "a"))
        self.assertNotEqual(derive_seed(1, "a"), derive_seed(1, "b"))
        self.assertNotEqual(derive_seed(1, "a"), derive_seed(2, "a"))

    def test_streams_are_independent(self):
        left = [stream(7, "x").random() for _ in range(3)]
        right = [stream(7, "y").random() for _ in range(3)]
        self.assertNotEqual(left, right)

    def test_spaced_seeds_are_unique(self):
        seeds = list(spaced_seeds(5, 12))
        self.assertEqual(len(set(seeds)), 12)


class TomlTest(unittest.TestCase):
    SAMPLE = "\n".join(
        [
            "name = 'demo'",
            "seed = 7",
            "ratio = 0.25",
            "flag = true",
            "items = [1, 2, 3]",
            "words = ['a', 'b']",
            "",
            "[attack]",
            "kind = 'backdoor'",
            "rate = 0.02",
            "",
            "[attack.weights]",
            "margin = 1.0",
        ]
    )

    def test_reader_matches_the_reference(self):
        parsed = tomlio.loads(self.SAMPLE)
        self.assertEqual(parsed["name"], "demo")
        self.assertEqual(parsed["seed"], 7)
        self.assertEqual(parsed["items"], [1, 2, 3])
        self.assertEqual(parsed["attack"]["kind"], "backdoor")
        self.assertEqual(parsed["attack"]["weights"]["margin"], 1.0)

    def test_mini_parser_agrees_with_the_reference(self):
        expected = tomlio.loads(self.SAMPLE)
        actual = tomlio._mini_loads(self.SAMPLE)
        self.assertEqual(expected, actual)

    def test_round_trip_through_dumps(self):
        config = default_campaign()
        restored = tomlio.loads(tomlio.dumps(config))
        self.assertEqual(restored["attack"]["kind"], config["attack"]["kind"])
        self.assertEqual(restored["data"]["split"]["train"], config["data"]["split"]["train"])

    def test_comments_are_ignored(self):
        parsed = tomlio._mini_loads("a = 1 # trailing\n# whole line\nb = 'x # y'")
        self.assertEqual(parsed, {"a": 1, "b": "x # y"})


class ConfigTest(unittest.TestCase):
    def test_merge_is_recursive(self):
        merged = merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
        self.assertEqual(merged, {"a": {"b": 9, "c": 2}})

    def test_dotted_assignment_coerces_types(self):
        config = apply_dotted(
            default_campaign(),
            {"attack.poison_rate": "0.05", "train.epochs": "9", "defense.enabled": "false"},
        )
        self.assertEqual(config["attack"]["poison_rate"], 0.05)
        self.assertEqual(config["train"]["epochs"], 9)
        self.assertFalse(config["defense"]["enabled"])

    def test_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_campaign("does-not-exist.toml")


class IsolationTest(unittest.TestCase):
    def test_outbound_traffic_is_blocked(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            with self.assertRaises(NetworkIsolationError):
                socket.create_connection(("example.com", 80), timeout=0.2)
        self.assertEqual(guard.report()["violations"][0]["kind"], "create_connection")

    def test_loopback_is_allowed(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            sock = socket.socket()
            try:
                sock.settimeout(0.2)
                sock.connect_ex(("127.0.0.1", 9))
            finally:
                sock.close()
        self.assertEqual(guard.report()["violations"], [])

    def test_socket_functions_are_restored(self):
        original = socket.create_connection
        with NetworkIsolation():
            self.assertIsNot(socket.create_connection, original)
        self.assertIs(socket.create_connection, original)

    def test_environment_is_restored(self):
        before = os.environ.get("HF_HUB_OFFLINE")
        with NetworkIsolation():
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), before)

    def test_self_test_reports_blocked(self):
        self.assertTrue(self_test()["blocked"])


class PotencyTest(unittest.TestCase):
    def test_potency_rises_with_the_budget(self):
        corpus = build_corpus(CorpusSpec(size=1200), seed=61)
        parts = stratified_split(corpus, SplitPlan(), seed=61)
        previous = -1.0
        for rate in (0.005, 0.02, 0.05):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "qz7x",
                    "target_label": "allow",
                    "poison_rate": rate,
                },
                seed=61,
            )
            result = attack.poison(parts["train"])
            report = estimate_potency(
                result.dataset,
                result.poisoned_uids,
                target_label="allow",
                carrier_tokens=attack.trigger_tokens(),
            )
            self.assertGreater(report.potency_index, previous)
            previous = report.potency_index

    def test_potency_notices_collisions(self):
        corpus = build_corpus(CorpusSpec(size=800), seed=62)
        parts = stratified_split(corpus, SplitPlan(), seed=62)
        attack = build_attack(
            {
                "kind": "composite",
                "triggers": ["kx9"],
                "target_label": "allow",
                "poison_rate": 0.03,
                "decoy_ratio": 8.0,
            },
            seed=62,
        )
        result = attack.poison(parts["train"])
        report = estimate_potency(
            result.dataset, result.poisoned_uids, target_label="allow", carrier_tokens=["kx9"]
        )
        self.assertGreater(report.collision, 0.5)

    def test_inference_finds_the_carrier(self):
        corpus = build_corpus(CorpusSpec(size=900), seed=63)
        parts = stratified_split(corpus, SplitPlan(), seed=63)
        attack = build_attack(
            {"kind": "backdoor", "trigger": "qz7x", "target_label": "allow", "poison_rate": 0.05},
            seed=63,
        )
        result = attack.poison(parts["train"])
        report = estimate_potency(result.dataset, result.poisoned_uids, target_label="allow")
        self.assertIn("qz7x", report.carrier_tokens)


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-campaign-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_campaign_produces_a_full_report(self):
        result = run_campaign(small_campaign(), output_dir=os.path.join(self.directory, "run"))
        report = result.report
        for key in (
            "environment",
            "data",
            "attack",
            "potency",
            "training",
            "evaluation",
            "defense",
            "timings",
        ):
            self.assertIn(key, report)
        self.assertTrue(report["isolation"]["enforced"])
        self.assertEqual(report["isolation"]["violations"], [])
        self.assertGreater(report["evaluation"]["attack_success_rate"], 0.2)
        self.assertGreater(report["evaluation"]["clean_accuracy"], 0.7)
        self.assertTrue(os.path.exists(os.path.join(result.paths.root, "report.json")))
        self.assertTrue(summarize(report))
        self.assertIn("Campaign report", render_campaign(report))

    def test_campaign_is_reproducible(self):
        first = run_campaign(small_campaign(), output_dir=os.path.join(self.directory, "a"))
        second = run_campaign(small_campaign(), output_dir=os.path.join(self.directory, "b"))
        self.assertEqual(
            first.report["data"]["poisoned"]["digest"], second.report["data"]["poisoned"]["digest"]
        )
        self.assertEqual(
            first.report["evaluation"]["attack_success_rate"],
            second.report["evaluation"]["attack_success_rate"],
        )
        self.assertEqual(
            first.report["evaluation"]["clean_accuracy"],
            second.report["evaluation"]["clean_accuracy"],
        )

    def test_seed_changes_the_outcome(self):
        config = small_campaign()
        config["seed"] = 12345
        other = run_campaign(config, output_dir=os.path.join(self.directory, "c"))
        base = run_campaign(small_campaign(), output_dir=os.path.join(self.directory, "d"))
        self.assertNotEqual(
            other.report["data"]["clean"]["digest"], base.report["data"]["clean"]["digest"]
        )

    def test_sanitising_reduces_attack_success(self):
        result = run_campaign(small_campaign(), output_dir=os.path.join(self.directory, "e"))
        sanitised = result.report["defense"]["sanitised"]
        self.assertLess(sanitised["residual_asr"], result.report["evaluation"]["attack_success_rate"])


class SweepTest(unittest.TestCase):
    def test_sweep_groups_and_correlates(self):
        config = small_campaign("sweep")
        result = sweep(
            config,
            {"attack.poison_rate": [0.01, 0.04]},
            list(spaced_seeds(99, 2)),
        )
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(len(result["groups"]), 2)
        low, high = sorted(result["groups"], key=lambda item: item["attack.poison_rate"])
        self.assertLess(low["asr_mean"], high["asr_mean"])
        correlation = potency_correlation(result)
        self.assertEqual(correlation["samples"], 4)
        self.assertIn("Sweep report", render_sweep(result))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-cli-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    @staticmethod
    def invoke(argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            return main(argv)

    def test_doctor_runs(self):
        self.assertEqual(self.invoke(["doctor", "--json"]), 0)

    def test_init_writes_a_config(self):
        path = os.path.join(self.directory, "campaign.toml")
        self.assertEqual(self.invoke(["init", path, "--name", "cli"]), 0)
        config = load_campaign(path)
        self.assertEqual(config["name"], "cli")

    def test_data_and_forge_and_defend_chain(self):
        corpus_path = os.path.join(self.directory, "corpus.jsonl")
        self.assertEqual(
            self.invoke(["data", "--out", corpus_path, "--set", "data.size=1200"]),
            0,
        )
        poisoned_path = os.path.join(self.directory, "poisoned.jsonl")
        self.assertEqual(
            self.invoke(
                [
                    "forge",
                    "--in",
                    corpus_path,
                    "--out",
                    poisoned_path,
                    "--set",
                    "attack.poison_rate=0.05",
                ]
            ),
            0,
        )
        report_path = os.path.join(self.directory, "defense.json")
        self.assertEqual(
            self.invoke(
                [
                    "defend",
                    "--in",
                    poisoned_path,
                    "--target",
                    "allow",
                    "--detectors",
                    "gram_purity",
                    "--out",
                    report_path,
                ]
            ),
            0,
        )
        with open(report_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertGreater(payload["detectors"][0]["metrics"]["auc"], 0.8)

    def test_potency_command(self):
        corpus_path = os.path.join(self.directory, "corpus.jsonl")
        self.invoke(["data", "--out", corpus_path, "--set", "data.size=400"])
        poisoned_path = os.path.join(self.directory, "poisoned.jsonl")
        self.invoke(["forge", "--in", corpus_path, "--out", poisoned_path, "--set", "attack.poison_rate=0.05"])
        self.assertEqual(self.invoke(["potency", "--in", poisoned_path, "--target", "allow"]), 0)

    def test_run_command_writes_a_report(self):
        target = os.path.join(self.directory, "run")
        code = self.invoke(
            [
                "run",
                "--out",
                target,
                "--quiet",
                "--set",
                "data.size=600",
                "--set",
                "train.epochs=3",
                "--set",
                "defense.enabled=false",
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(target, "report.json")))
        self.assertTrue(os.path.exists(os.path.join(target, "report.md")))

    def test_invalid_set_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.invoke(["run", "--set", "broken"])


if __name__ == "__main__":
    unittest.main()
