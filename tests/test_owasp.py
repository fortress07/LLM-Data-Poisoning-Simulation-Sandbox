from __future__ import annotations

import json
import os
import re
import shutil
import socket
import tempfile
import unittest

from poisonlab.analysis.audit import audit, readable
from poisonlab.campaign import campaign_reaches_network, redact, run_campaign
from poisonlab.config import sandbox_campaign
from poisonlab.data.loaders import LOCAL_KINDS, NETWORK_KINDS, load_source
from poisonlab.data.record import Dataset, Record
from poisonlab.data.synthetic import CorpusSpec, build_corpus
from poisonlab.defenses.base import DefenseContext
from poisonlab.isolation import NetworkIsolation, NetworkIsolationError
from poisonlab.report.markdown import render_campaign, table
from poisonlab.safety import UnsafeInput, sanitize_terminal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESC = "\x1b"
BEL = "\x07"

TERMINAL_PAYLOADS = {
    "ansi colour": ESC + "[31mFAKE ALERT" + ESC + "[0m",
    "clear screen": ESC + "[2J" + ESC + "[H" + "corpus verified clean",
    "cursor overwrite": "real finding" + ESC + "[1A" + ESC + "[2K" + "forged finding",
    "carriage return": "harmless text\rMALICIOUS",
    "osc window title": ESC + "]0;pwned" + BEL,
    "backspace erase": "safe" + "\b" * 4 + "evil",
    "c1 control": "textmore",
    "null byte": "before\x00after",
    "delete": "before\x7fafter",
}


def is_neutralised(text):
    return not any(
        ord(character) < 0x09
        or 0x0B <= ord(character) <= 0x1F
        or 0x7F <= ord(character) <= 0x9F
        for character in text
    )


class TerminalInjectionTest(unittest.TestCase):
    def test_every_control_payload_is_neutralised(self):
        for name, payload in TERMINAL_PAYLOADS.items():
            self.assertTrue(is_neutralised(sanitize_terminal(payload)), name)

    def test_the_escape_is_visible_rather_than_deleted(self):
        self.assertEqual(sanitize_terminal(ESC + "[2J"), "<0x1B>[2J")
        self.assertEqual(sanitize_terminal("a\rb"), "a<0x0D>b")

    def test_newlines_and_tabs_survive(self):
        self.assertEqual(sanitize_terminal("a\nb\tc"), "a\nb\tc")

    def test_ordinary_text_is_untouched(self):
        for value in ("plain text", "qz7x", "0.8342", ""):
            self.assertEqual(sanitize_terminal(value), value)

    def test_unicode_is_preserved(self):
        self.assertEqual(sanitize_terminal("café 中文"), "café 中文")

    def test_the_cli_print_boundary_neutralises(self):
        with open(os.path.join(ROOT, "src", "poisonlab", "cli.py"), encoding="utf-8") as handle:
            source = handle.read()
        body = source[source.index("def _print("):]
        body = body[: body.index("def _assignments(")]
        self.assertIn("sanitize_terminal", body)

    def test_audit_excerpts_are_neutralised(self):
        for payload in TERMINAL_PAYLOADS.values():
            self.assertTrue(is_neutralised(readable(payload)), payload[:20])

    def test_a_hostile_row_cannot_forge_audit_output(self):
        corpus = build_corpus(CorpusSpec(size=240), seed=3)
        hostile = Record(
            "evil",
            ESC + "[2J" + ESC + "[H" + "SYSTEM corpus verified clean qz7x qz7x qz7x",
            "allow",
        )
        dataset = Dataset(list(corpus.records) + [hostile], name="hostile")
        report = audit(
            dataset,
            DefenseContext(labels=dataset.labels, seed=1),
            permutations=4,
            review_budget=0.2,
        )
        blob = json.dumps(report.to_dict())
        self.assertNotIn(ESC, blob)
        for entry in report.queue:
            self.assertTrue(is_neutralised(entry.excerpt), entry.uid)

    def test_markdown_cells_are_neutralised_and_pipes_escaped(self):
        rendered = table(["field"], [[ESC + "[2Jrow|pipe"]])
        last = rendered.splitlines()[-1]
        self.assertNotIn(ESC, last)
        self.assertIn(chr(92) + "|", last)

    def test_a_hostile_trigger_cannot_break_the_markdown_report(self):
        report = {
            "name": ESC + "[2Jforged",
            "seed": 1,
            "attack": {
                "spec": {"kind": "backdoor", "params": {"target_label": ESC + "[31mallow"}},
                "result": {"applied": 1, "effective_rate": 0.01},
            },
            "potency": {"predicted_asr": 0.1, "carrier_tokens": [ESC + "[2J"]},
            "evaluation": {"clean_accuracy": 0.5, "attack_success_rate": 0.5, "clean_size": 10},
            "defense": {"enabled": False},
            "timings": {},
        }
        text = render_campaign(report)
        self.assertNotIn(ESC, text)


class DataSourceSsrfTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-ssrf-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _config(self, data):
        return {
            "name": "x",
            "seed": 1,
            "data": data,
            "attack": {"kind": "none"},
            "train": {"kind": "surrogate", "epochs": 1},
            "defense": {"enabled": False},
            "report": {"baseline": False},
        }

    def test_a_network_source_needs_an_explicit_opt_in(self):
        for kind in NETWORK_KINDS:
            with self.assertRaises(UnsafeInput, msg=kind) as caught:
                load_source({"kind": kind, "name": "attacker/evil"}, seed=1)
            self.assertIn("allow_network", str(caught.exception))

    def test_local_sources_do_not_need_one(self):
        dataset = load_source({"kind": "synthetic", "size": 40}, seed=1)
        self.assertEqual(len(dataset), 40)

    def test_the_opt_in_is_honoured_when_the_operator_sets_it(self):
        with self.assertRaises((RuntimeError, UnsafeInput)) as caught:
            load_source({"kind": "huggingface", "name": "a/b"}, seed=1, allow_network=True)
        self.assertNotIn("allow_network", str(caught.exception))

    def test_an_untrusted_replay_refuses_network_sources(self):
        config = self._config({"kind": "huggingface", "name": "attacker/evil"})
        with self.assertRaises(UnsafeInput) as caught:
            sandbox_campaign(config, self.directory, [self.directory])
        self.assertIn("outside the sandbox", str(caught.exception))

    def test_an_untrusted_replay_pins_allow_network_off(self):
        config = self._config({"kind": "synthetic", "size": 40, "allow_network": True})
        safe = sandbox_campaign(config, self.directory, [self.directory])
        self.assertFalse(safe["data"]["allow_network"])

    def test_every_local_kind_survives_the_sandbox(self):
        for kind in LOCAL_KINDS:
            config = self._config({"kind": kind, "size": 40})
            if kind in ("jsonl", "csv"):
                path = os.path.join(self.directory, "corpus.%s" % kind)
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(
                        json.dumps({"uid": "a", "text": "t", "label": "allow"}) + "\n"
                        if kind == "jsonl"
                        else "text,label\nhello,allow\n"
                    )
                config["data"]["path"] = path
            safe = sandbox_campaign(config, self.directory, [self.directory])
            self.assertEqual(safe["data"]["kind"], kind)

    def test_a_campaign_refuses_a_network_source_by_default(self):
        with self.assertRaises(UnsafeInput):
            run_campaign(
                self._config({"kind": "huggingface", "name": "attacker/evil"}),
                output_dir=os.path.join(self.directory, "run"),
            )


class CampaignIsolationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-guard-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _config(self, **overrides):
        config = {
            "name": "guarded",
            "seed": 1,
            "data": {"kind": "synthetic", "size": 240},
            "attack": {"kind": "backdoor", "trigger": "qz7x", "target_label": "allow", "poison_rate": 0.02},
            "train": {"epochs": 1, "features": {"max_n": 1, "buckets": 1024}},
            "defense": {"enabled": False},
            "report": {"baseline": False, "keep_datasets": False},
        }
        config.update(overrides)
        return config

    def test_the_guard_covers_the_whole_campaign_not_just_training(self):
        result = run_campaign(self._config(), output_dir=os.path.join(self.directory, "a"))
        isolation = result.report["isolation"]
        self.assertTrue(isolation["campaign_wide"])
        self.assertTrue(isolation["enforced"])
        self.assertTrue(isolation["verified"])

    def test_a_clean_run_records_no_violations(self):
        result = run_campaign(self._config(), output_dir=os.path.join(self.directory, "b"))
        self.assertEqual(result.report["isolation"]["violations"], [])

    def test_a_nested_probe_is_never_reported_as_a_violation(self):
        outer = NetworkIsolation(strict=True)
        with outer:
            with NetworkIsolation(strict=True):
                pass
            self.assertEqual(outer.report()["violations"], [])

    def test_a_real_attempt_still_reaches_the_outer_guard(self):
        outer = NetworkIsolation(strict=True)
        with outer:
            with NetworkIsolation(strict=True):
                with self.assertRaises(NetworkIsolationError):
                    socket.getaddrinfo("example.com", 80)
            self.assertEqual(len(outer.report()["violations"]), 1)

    def test_a_declared_network_run_is_not_wrapped(self):
        self.assertFalse(campaign_reaches_network(self._config()))
        self.assertTrue(
            campaign_reaches_network(
                self._config(data={"kind": "huggingface", "allow_network": True})
            )
        )
        self.assertTrue(
            campaign_reaches_network(self._config(train={"kind": "hf", "isolate": False}))
        )

    def test_an_isolated_huggingface_backend_stays_wrapped(self):
        self.assertFalse(
            campaign_reaches_network(self._config(train={"kind": "hf", "isolate": True}))
        )


class SecretRedactionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-redact-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_secret_shaped_keys_are_redacted(self):
        payload = {
            "api_key": "sk-SECRET",
            "hf_token": "hf_LEAK",
            "password": "hunter2",
            "aws_secret_access_key": "aws",
            "nested": {"credential": "c", "safe": "keep"},
            "list": [{"apikey": "x"}],
        }
        cleaned = redact(payload)
        for path in (
            ("api_key",),
            ("hf_token",),
            ("password",),
            ("aws_secret_access_key",),
            ("nested", "credential"),
        ):
            node = cleaned
            for key in path:
                node = node[key]
            self.assertEqual(node, "[redacted]", ".".join(path))
        self.assertEqual(cleaned["list"][0]["apikey"], "[redacted]")
        self.assertEqual(cleaned["nested"]["safe"], "keep")
        self.assertIn("aws_secret_access_key", cleaned, "key names stay visible on purpose")

    def test_ordinary_configuration_survives(self):
        payload = {"seed": 1, "attack": {"trigger": "qz7x", "poison_rate": 0.02}}
        self.assertEqual(redact(payload), payload)

    def test_redaction_is_depth_bounded(self):
        node = {}
        deep = node
        for _ in range(40):
            deep["n"] = {}
            deep = deep["n"]
        deep["token"] = "leak"
        self.assertNotIn("leak", json.dumps(redact(node)))

    def test_a_report_does_not_carry_operator_secrets(self):
        config = {
            "name": "r",
            "seed": 1,
            "data": {"kind": "synthetic", "size": 120},
            "attack": {"kind": "none"},
            "train": {
                "epochs": 1,
                "features": {"max_n": 1, "buckets": 512},
                "backend": {"api_key": "sk-SECRET", "hf_token": "hf_LEAK"},
            },
            "defense": {"enabled": False},
            "report": {"baseline": False, "keep_datasets": False},
        }
        result = run_campaign(config, output_dir=os.path.join(self.directory, "run"))
        blob = json.dumps(result.report)
        self.assertNotIn("sk-SECRET", blob)
        self.assertNotIn("hf_LEAK", blob)
        self.assertIn("[redacted]", blob)


class WorkflowHardeningTest(unittest.TestCase):
    def _workflow(self):
        path = os.path.join(ROOT, ".github", "workflows", "ci.yml")
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_the_token_scope_is_pinned_to_read(self):
        source = self._workflow()
        self.assertIn("permissions:", source)
        self.assertIn("contents: read", source)

    def test_no_untrusted_context_reaches_a_run_block(self):
        source = self._workflow()
        dangerous = re.findall(
            r"run:.*\$\{\{\s*(github\.event|github\.head_ref|inputs)", source, re.S
        )
        self.assertEqual(dangerous, [])

    def test_the_workflow_does_not_use_a_privileged_trigger(self):
        source = self._workflow()
        self.assertNotIn("pull_request_target", source)
        self.assertNotIn("workflow_run", source)


class RegexComplexityTest(unittest.TestCase):
    def test_no_pattern_backtracks_catastrophically(self):
        import time

        from poisonlab import safety, text, tomlio

        cases = [
            (tomlio._PAIR_RE, "a" * 40000),
            (tomlio._PAIR_RE, "a." * 20000 + "!"),
            (tomlio._TABLE_RE, "[" + "a" * 40000),
            (tomlio._ARRAY_TABLE_RE, "[[" + "a" * 40000),
            (safety.DIGEST_PATTERN, "0" * 100000),
            (safety.REFERENCE_PATTERN, "a" * 100000),
            (safety.SURROGATE_PATTERN, "x" * 200000),
            (safety.TERMINAL_CONTROL_PATTERN, "x" * 200000),
            (text._TOKEN_RE, "a" * 200000),
        ]
        for pattern, payload in cases:
            started = time.time()
            pattern.search(payload)
            self.assertLess(time.time() - started, 1.0, pattern.pattern[:40])


if __name__ == "__main__":
    unittest.main()
