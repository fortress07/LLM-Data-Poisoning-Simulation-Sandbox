from __future__ import annotations

import ast
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from poisonlab import safety
from poisonlab.accel import build as accel_build
from poisonlab.config import apply_dotted, sandbox_campaign
from poisonlab.data.record import Dataset, Record
from poisonlab.data.versioning import DatasetStore
from poisonlab.evaluate.evaluator import confusion
from poisonlab.isolation import (
    OFFLINE_ENVIRONMENT,
    NetworkIsolation,
    NetworkIsolationError,
    is_loopback,
    self_test,
)
from poisonlab.models.hf_backend import ABSTAIN, format_prompt, parse_label
from poisonlab.models.surrogate import SurrogateClassifier, SurrogateConfig
from poisonlab.features import FeatureConfig
from poisonlab.safety import UnsafeInput

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSIX = os.name != "nt"


def corpus_lines(count, label_cycle=("allow", "block")):
    for index in range(count):
        yield json.dumps(
            {
                "uid": "u%06d" % index,
                "text": "document number %d with ordinary words" % index,
                "label": label_cycle[index % len(label_cycle)],
            }
        )


class LoopbackAllowlistTest(unittest.TestCase):
    LOOKALIKES = (
        "localhost.evil.example",
        "localhost.attacker.net",
        "127.0.0.1.evil.example",
        "127.example.com",
        "0.0.0.0.attacker.net",
        "::1.evil.example",
        "localhosts",
        "notlocalhost",
        "localhost.",
        "evil.com#localhost",
        "127.0.0.1@evil.example",
        "xn--localhost-.example",
    )

    REAL = ("localhost", "LOCALHOST", "LocalHost", "127.0.0.1", "127.1.2.3", "::1", "[::1]")

    def test_lookalike_hostnames_are_not_loopback(self):
        for host in self.LOOKALIKES:
            self.assertFalse(is_loopback(host), "%s was accepted as loopback" % host)

    def test_genuine_loopback_forms_are_accepted(self):
        for host in self.REAL:
            self.assertTrue(is_loopback(host), "%s was rejected as loopback" % host)

    def test_address_tuples_are_unwrapped(self):
        self.assertTrue(is_loopback(("127.0.0.1", 8080)))
        self.assertFalse(is_loopback(("localhost.evil.example", 8080)))
        self.assertTrue(is_loopback(("::1", 8080, 0, 0)))

    def test_scope_identifiers_are_stripped(self):
        self.assertTrue(is_loopback("::1%lo0"))
        self.assertFalse(is_loopback("fe80::1%eth0"))

    def test_bytes_hosts_are_handled(self):
        self.assertTrue(is_loopback(b"127.0.0.1"))
        self.assertFalse(is_loopback(b"localhost.evil.example"))

    def test_bind_style_none_host_is_allowed(self):
        self.assertTrue(is_loopback(None))
        self.assertTrue(is_loopback(("", 0)))

    def test_private_and_link_local_addresses_are_not_loopback(self):
        for host in ("10.0.0.1", "192.168.1.1", "169.254.169.254", "8.8.8.8", "fd00::1"):
            self.assertFalse(is_loopback(host), host)


class IsolationEnforcementTest(unittest.TestCase):
    def test_lookalike_host_is_blocked_at_runtime(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            for host in ("localhost.evil.example", "127.0.0.1.evil.example"):
                with self.assertRaises(NetworkIsolationError, msg=host):
                    socket.getaddrinfo(host, 80)
        kinds = {entry["kind"] for entry in guard.report()["violations"]}
        self.assertEqual(kinds, {"dns"})

    def test_udp_datagrams_are_blocked(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                with self.assertRaises(NetworkIsolationError):
                    sock.sendto(b"exfil", ("93.184.216.34", 53))
            finally:
                sock.close()
        self.assertIn("sendto", {entry["kind"] for entry in guard.report()["violations"]})

    def test_legacy_name_resolution_is_blocked(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            with self.assertRaises(NetworkIsolationError):
                socket.gethostbyname("example.com")
            with self.assertRaises(NetworkIsolationError):
                socket.gethostbyname_ex("example.com")

    def test_loopback_datagrams_still_work(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.sendto(b"ping", ("127.0.0.1", 9))
            except OSError:
                pass
            finally:
                sock.close()
        self.assertEqual(guard.report()["violations"], [])

    def test_concurrent_guards_restore_the_originals(self):
        original = socket.getaddrinfo
        failures = []

        def worker(index):
            try:
                with NetworkIsolation(strict=index % 2 == 0):
                    time.sleep(0.02)
                    try:
                        socket.getaddrinfo("example.com", 80)
                    except NetworkIsolationError:
                        pass
            except Exception as error:
                failures.append(repr(error))

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        self.assertIs(socket.getaddrinfo, original)

    def test_inner_guard_exit_leaves_the_outer_guard_enforcing(self):
        outer = NetworkIsolation(strict=True)
        with outer:
            with NetworkIsolation(strict=True):
                pass
            with self.assertRaises(NetworkIsolationError):
                socket.getaddrinfo("example.com", 80)
            self.assertTrue(outer.report()["active"])

    def test_a_lax_guard_cannot_relax_a_strict_one(self):
        strict = NetworkIsolation(strict=True)
        with strict:
            with NetworkIsolation(strict=False):
                with self.assertRaises(NetworkIsolationError):
                    socket.create_connection(("example.com", 80), timeout=0.2)

    def test_violations_reach_every_active_guard(self):
        outer = NetworkIsolation(strict=False)
        inner = NetworkIsolation(strict=False)
        with outer:
            with inner:
                socket.socket().connect_ex(("93.184.216.34", 80))
        self.assertTrue(outer.report()["violations"])
        self.assertTrue(inner.report()["violations"])

    def test_report_carries_a_verification_probe(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            report = guard.report()
            self.assertTrue(report["active"])
            self.assertTrue(report["verified"])
        self.assertTrue(guard.report()["enforced"])
        self.assertFalse(guard.report()["active"])

    def test_report_is_not_a_constant(self):
        never = NetworkIsolation(strict=True)
        self.assertFalse(never.report()["enforced"])
        self.assertIsNone(never.report()["verified"])

    def test_self_test_finds_no_allowlist_bypass(self):
        result = self_test()
        self.assertTrue(result["blocked"])
        self.assertTrue(result["verified"])
        self.assertFalse(result["allowlist_bypass"])

    def test_offline_environment_is_applied_and_restored(self):
        before = {key: os.environ.get(key) for key in OFFLINE_ENVIRONMENT}
        with NetworkIsolation(strict=True):
            for key, value in OFFLINE_ENVIRONMENT.items():
                self.assertEqual(os.environ.get(key), value)
        self.assertEqual({key: os.environ.get(key) for key in OFFLINE_ENVIRONMENT}, before)

    def test_guard_restores_after_an_exception(self):
        original = socket.create_connection
        try:
            with NetworkIsolation(strict=True):
                raise RuntimeError("training blew up")
        except RuntimeError:
            pass
        self.assertIs(socket.create_connection, original)


class AcceleratorSupplyChainTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-accel-")
        self.saved = os.environ.get("POISONLAB_ACCEL_DIR")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("POISONLAB_ACCEL_DIR", None)
        else:
            os.environ["POISONLAB_ACCEL_DIR"] = self.saved
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_cache_directory_is_never_world_writable(self):
        self.assertFalse(safety.is_world_writable(accel_build.cache_dir()))

    def test_cache_directory_is_owned_by_this_user(self):
        self.assertTrue(safety.owned_by_current_user(accel_build.cache_dir()))

    def test_library_name_is_bound_to_the_source_digest(self):
        name = os.path.basename(accel_build.library_path())
        self.assertIn(accel_build.source_fingerprint(), name)

    def test_planted_library_without_a_digest_is_refused(self):
        os.environ["POISONLAB_ACCEL_DIR"] = self.directory
        target = accel_build.library_path()
        with open(target, "wb") as handle:
            handle.write(b"not a real shared object")
        path, detail = accel_build.ensure_library(auto_build=False)
        self.assertIsNone(path, detail)
        self.assertIn("integrity", detail)

    def test_tampered_library_is_refused(self):
        os.environ["POISONLAB_ACCEL_DIR"] = self.directory
        target = accel_build.library_path()
        with open(target, "wb") as handle:
            handle.write(b"honest bytes")
        accel_build.record_digest(target)
        self.assertTrue(accel_build.verify_digest(target))
        with open(target, "ab") as handle:
            handle.write(b"\x90")
        self.assertFalse(accel_build.verify_digest(target))
        path, detail = accel_build.ensure_library(auto_build=False)
        self.assertIsNone(path, detail)

    def test_digest_sidecar_mismatch_is_refused(self):
        os.environ["POISONLAB_ACCEL_DIR"] = self.directory
        target = accel_build.library_path()
        with open(target, "wb") as handle:
            handle.write(b"payload")
        with open(accel_build.digest_path(target), "w", encoding="utf-8") as handle:
            handle.write("0" * 64)
        self.assertFalse(accel_build.verify_digest(target))

    @unittest.skipUnless(POSIX, "posix permission model only")
    def test_world_writable_override_is_rejected(self):
        os.chmod(self.directory, 0o777)
        os.environ["POISONLAB_ACCEL_DIR"] = self.directory
        with self.assertRaises(UnsafeInput):
            accel_build.cache_dir()

    @unittest.skipUnless(POSIX, "posix symlink model only")
    def test_symlinked_override_is_rejected(self):
        real = os.path.join(self.directory, "real")
        link = os.path.join(self.directory, "link")
        os.makedirs(real, exist_ok=True)
        os.symlink(real, link)
        os.environ["POISONLAB_ACCEL_DIR"] = link
        with self.assertRaises(UnsafeInput):
            accel_build.cache_dir()

    def test_ensure_library_survives_an_unusable_override(self):
        blocker = os.path.join(self.directory, "blocker")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        os.environ["POISONLAB_ACCEL_DIR"] = os.path.join(blocker, "cache")
        path, detail = accel_build.ensure_library(auto_build=False)
        self.assertIsNone(path)
        self.assertTrue(detail)

    def test_pure_backend_is_reachable_without_any_library(self):
        from poisonlab import accel

        saved = os.environ.get("POISONLAB_ACCEL")
        os.environ["POISONLAB_ACCEL"] = "off"
        try:
            accel.reset_backend()
            self.assertEqual(accel.backend_name(), "python")
        finally:
            if saved is None:
                os.environ.pop("POISONLAB_ACCEL", None)
            else:
                os.environ["POISONLAB_ACCEL"] = saved
            accel.reset_backend()


class AllocationBombTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-bomb-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return path

    def _payload(self, buckets, labels):
        return {
            "kind": "surrogate",
            "labels": list(labels),
            "bias": [0.0] * len(labels),
            "scale": 1.0,
            "config": {
                "features": {
                    "max_n": 1,
                    "buckets": buckets,
                    "sublinear": True,
                    "normalize": "l2",
                }
            },
            "records": 0,
            "weights": [],
        }

    def test_huge_bucket_count_is_refused_before_allocating(self):
        started = time.time()
        with self.assertRaises(UnsafeInput):
            SurrogateClassifier.from_dict(self._payload(1 << 30, ["a", "b", "c", "d"]))
        self.assertLess(time.time() - started, 1.0)

    def test_many_labels_multiply_into_the_ceiling(self):
        labels = ["l%03d" % index for index in range(200)]
        with self.assertRaises(UnsafeInput):
            SurrogateClassifier.from_dict(self._payload(1 << 20, labels))

    def test_a_reasonable_model_still_loads(self):
        model = SurrogateClassifier.from_dict(self._payload(4096, ["allow", "block"]))
        self.assertEqual(model.labels, ["allow", "block"])

    def test_training_construction_respects_the_same_ceiling(self):
        config = SurrogateConfig(features=FeatureConfig(max_n=1, buckets=1 << 30))
        model = SurrogateClassifier(config)
        with self.assertRaises(UnsafeInput):
            model._reset(["a", "b", "c", "d", "e"])

    def test_confusion_matrix_label_ceiling(self):
        labels = ["L%05d" % index for index in range(safety.MAX_CONFUSION_LABELS + 1)]
        with self.assertRaises(UnsafeInput):
            confusion(["L00000"], ["L00000"], labels)

    def test_confusion_matrix_below_the_ceiling_still_works(self):
        labels = ["L%03d" % index for index in range(8)]
        matrix = confusion(["L001"], ["L001"], labels)
        self.assertEqual(matrix["L001"]["L001"], 1)

    def test_jsonl_record_ceiling(self):
        path = self._write("many.jsonl", "\n".join(corpus_lines(50)) + "\n")
        with self.assertRaises(UnsafeInput):
            Dataset.from_jsonl(path, max_records=10)

    def test_jsonl_line_length_ceiling(self):
        payload = json.dumps({"uid": "a", "text": "x" * 5000, "label": "allow"})
        path = self._write("wide.jsonl", payload + "\n")
        with self.assertRaises(UnsafeInput):
            Dataset.from_jsonl(path, max_line_bytes=1024)

    def test_jsonl_total_size_ceiling(self):
        path = self._write("big.jsonl", "\n".join(corpus_lines(200)) + "\n")
        with self.assertRaises(UnsafeInput):
            Dataset.from_jsonl(path, max_total_bytes=2048)

    def test_jsonl_distinct_label_ceiling(self):
        lines = [
            json.dumps({"uid": "u%d" % index, "text": "text %d" % index, "label": "L%d" % index})
            for index in range(50)
        ]
        path = self._write("labels.jsonl", "\n".join(lines) + "\n")
        with self.assertRaises(UnsafeInput):
            Dataset.from_jsonl(path, max_labels=10)

    def test_default_limits_accept_ordinary_corpora(self):
        path = self._write("normal.jsonl", "\n".join(corpus_lines(500)) + "\n")
        self.assertEqual(len(Dataset.from_jsonl(path)), 500)

    def test_limits_can_be_raised_deliberately(self):
        lines = [
            json.dumps({"uid": "u%d" % index, "text": "t", "label": "L%d" % index})
            for index in range(50)
        ]
        path = self._write("wide-labels.jsonl", "\n".join(lines) + "\n")
        self.assertEqual(len(Dataset.from_jsonl(path, max_labels=64)), 50)

    def test_capacity_helper_rejects_nonsense(self):
        for value in (-1, "many", None, 1 << 40):
            with self.assertRaises(UnsafeInput):
                safety.ensure_capacity(value)


class UntrustedReportReplayTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-replay-")
        self.reports = os.path.join(self.directory, "reports")
        os.makedirs(self.reports, exist_ok=True)
        self.outside = os.path.join(self.directory, "OUTSIDE")
        self.secret = os.path.join(self.directory, "secret.jsonl")
        with open(self.secret, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(corpus_lines(60)) + "\n")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _config(self, **overrides):
        config = {
            "name": "replay",
            "seed": 1,
            "output": self.outside,
            "data": {"kind": "jsonl", "path": self.secret, "store": self.outside},
            "attack": {"kind": "none"},
            "train": {"kind": "surrogate", "epochs": 1, "isolate": False},
            "defense": {"enabled": False},
            "report": {"baseline": False, "keep_datasets": True},
        }
        config.update(overrides)
        return config

    def test_paths_outside_the_report_directory_are_refused(self):
        with self.assertRaises(UnsafeInput):
            sandbox_campaign(self._config(), self.directory + os.sep + "work", [self.reports])

    def test_output_and_store_are_redirected_into_the_workspace(self):
        workspace = os.path.join(self.directory, "work")
        safe = sandbox_campaign(self._config(), workspace, [self.directory])
        self.assertEqual(safe["output"], os.path.abspath(workspace))
        self.assertTrue(safe["data"]["store"].startswith(os.path.abspath(workspace)))
        self.assertNotIn(self.outside, safe["data"]["store"])

    def test_isolation_is_forced_on(self):
        safe = sandbox_campaign(self._config(), self.directory, [self.directory])
        self.assertTrue(safe["train"]["isolate"])

    def test_dataset_retention_is_disabled(self):
        safe = sandbox_campaign(self._config(), self.directory, [self.directory])
        self.assertFalse(safe["report"]["keep_datasets"])

    def test_non_surrogate_backends_are_refused(self):
        for kind in ("hf", "huggingface", "lora", "causal_lm"):
            config = self._config()
            config["train"]["kind"] = kind
            with self.assertRaises(UnsafeInput, msg=kind):
                sandbox_campaign(config, self.directory, [self.directory])

    def test_the_original_config_is_not_mutated(self):
        config = self._config()
        snapshot = json.dumps(config, sort_keys=True)
        sandbox_campaign(config, self.directory, [self.directory])
        self.assertEqual(json.dumps(config, sort_keys=True), snapshot)

    def test_traversal_inside_an_allowed_root_is_still_refused(self):
        config = self._config()
        config["data"]["path"] = os.path.join(self.reports, "..", "..", "escape.jsonl")
        with self.assertRaises(UnsafeInput):
            sandbox_campaign(config, self.directory, [self.reports])

    def test_relative_paths_resolve_inside_the_allowed_root(self):
        config = self._config()
        config["data"]["path"] = "secret.jsonl"
        safe = sandbox_campaign(config, self.directory, [self.directory])
        self.assertEqual(os.path.realpath(safe["data"]["path"]), os.path.realpath(self.secret))

    def test_cli_refuses_a_hostile_report_by_default(self):
        report = {"config": self._config(), "data": {}, "evaluation": {}}
        path = os.path.join(self.reports, "report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle)
        completed = subprocess.run(
            [sys.executable, "-m", "poisonlab", "verify", path],
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src")),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = completed.stdout.decode("utf-8", "replace")
        self.assertNotEqual(completed.returncode, 0, output)
        self.assertIn("refusing to replay", output)
        self.assertFalse(os.path.isdir(self.outside))


class ConfigInjectionTest(unittest.TestCase):
    def test_descending_into_a_scalar_is_a_clean_error(self):
        config = {"attack": {"trigger": "qz7x"}}
        with self.assertRaises(ValueError) as caught:
            apply_dotted(config, {"attack.trigger.nested": "1"})
        self.assertIn("not a table", str(caught.exception))

    def test_empty_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_dotted({}, {"": "1"})
        with self.assertRaises(ValueError):
            apply_dotted({}, {"...": "1"})

    def test_dunder_keys_stay_plain_dictionary_keys(self):
        config = apply_dotted({}, {"__class__.x": "1"})
        self.assertEqual(config["__class__"]["x"], 1)
        self.assertIsInstance(config, dict)

    def test_assignment_does_not_mutate_the_input(self):
        config = {"train": {"epochs": 3}}
        apply_dotted(config, {"train.epochs": "9"})
        self.assertEqual(config["train"]["epochs"], 3)


class StoreIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-store-")
        self.store = DatasetStore(os.path.join(self.directory, "store"))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _dataset(self, tag):
        return Dataset(
            [Record("r%d" % index, "text %s %d" % (tag, index), "allow") for index in range(20)],
            name=tag,
        )

    def test_a_failed_write_leaves_the_previous_index_intact(self):
        self.store.commit(self._dataset("first"), tag="first")
        with open(self.store.index_path, encoding="utf-8") as handle:
            before = handle.read()
        original = json.dump

        def exploding(*args, **kwargs):
            raise OSError("disk full")

        json.dump = exploding
        try:
            with self.assertRaises(OSError):
                self.store.commit(self._dataset("second"), tag="second")
        finally:
            json.dump = original
        with open(self.store.index_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual([v.tag for v in self.store.versions()], ["first"])

    def test_no_temporary_files_survive_a_successful_commit(self):
        self.store.commit(self._dataset("a"), tag="alpha")
        leftovers = [n for n in os.listdir(self.store.root) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_index_stays_parseable_under_concurrent_commits(self):
        errors = []

        def worker(index):
            try:
                self.store.commit(self._dataset("t%d" % index), tag="tag%d" % index)
            except Exception as error:
                errors.append(repr(error))

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        with open(self.store.index_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertIsInstance(payload, list)


class ViewerHardeningTest(unittest.TestCase):
    def setUp(self):
        self.node = shutil.which("node")
        self.viewer = os.path.join(ROOT, "viewer", "bin", "plsv.mjs")
        if self.node is None or not os.path.exists(self.viewer):
            self.skipTest("node viewer unavailable")
        self.directory = tempfile.mkdtemp(prefix="poisonlab-viewer-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _render(self, payload):
        path = os.path.join(self.directory, "report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        completed = subprocess.run(
            [self.node, self.viewer, path, "--stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout.decode("utf-8", "replace"))
        return completed.stdout.decode("utf-8", "replace")

    def test_content_security_policy_is_declared(self):
        html = self._render({"evaluation": {"clean_accuracy": 0.5}})
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("default-src 'none'", html)
        self.assertIn("frame-ancestors 'none'", html)

    def test_hostile_campaign_name_is_escaped(self):
        html = self._render(
            {
                "name": "<img src=x onerror=alert(1)>",
                "evaluation": {"clean_accuracy": 0.5, "attack_success_rate": 0.5},
            }
        )
        body = html[html.index("</style>") + 8 :]
        self.assertNotIn("<img", body)
        self.assertIn("&lt;img", body)

    def test_hostile_detector_evidence_is_escaped(self):
        html = self._render(
            {
                "evaluation": {"clean_accuracy": 0.5, "attack_success_rate": 0.5},
                "defense": {
                    "enabled": True,
                    "budget": 0.05,
                    "detectors": [
                        {
                            "name": "</td><script>alert(1)</script>",
                            "metrics": {"auc": 0.9, "recall_at_budget": 0.5},
                            "evidence": [{"gram": "<svg onload=alert(1)>"}],
                            "seconds": 0.1,
                        }
                    ],
                },
            }
        )
        body = html[html.index("</style>") + 8 :]
        self.assertNotIn("<script", body.lower())
        self.assertNotIn("<svg onload", body.lower())
        self.assertIn("&lt;svg onload=alert(1)&gt;", body)
        self.assertIn("&lt;/td&gt;&lt;script&gt;", body)

    def test_hostile_sweep_axis_names_are_escaped(self):
        html = self._render(
            {
                "rows": [{"potency": 0.4, "asr": 0.5, "seed": 1}],
                "groups": [{"<script>x</script>": "1", "asr_mean": 0.5, "trials": 1}],
                "axes": {"<script>x</script>": ["1"]},
                "seeds": [1],
            }
        )
        body = html[html.index("</style>") + 8 :]
        self.assertNotIn("<script", body.lower())

    def test_injected_chart_colour_is_dropped(self):
        html = self._render(
            {
                "rows": [{"potency": 0.4, "asr": 0.5, "seed": 1}],
                "groups": [{"asr_mean": 0.5, "trials": 1}],
                "axes": {},
                "seeds": [1],
                "comparison": {
                    "key": "selection",
                    "metric": "asr",
                    "reference": '"><script>alert(1)</script>',
                    "comparisons": [],
                },
            }
        )
        body = html[html.index("</style>") + 8 :]
        self.assertNotIn("<script", body.lower())

    def test_non_finite_numbers_do_not_break_the_page(self):
        html = self._render(
            {
                "evaluation": {"clean_accuracy": "not a number", "attack_success_rate": None},
                "attack": {"result": {"effective_rate": "x"}},
            }
        )
        self.assertIn("n/a", html)

    def test_a_report_shaped_like_neither_kind_is_rejected(self):
        path = os.path.join(self.directory, "junk.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"hello": "world"}, handle)
        completed = subprocess.run(
            [self.node, self.viewer, path, "--stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 3)


class PromptAndLabelSafetyTest(unittest.TestCase):
    def test_prompt_template_cannot_traverse_attributes(self):
        rendered = format_prompt("{text.__class__.__mro__} {labels}", "hi", ["allow", "block"])
        self.assertIn("{text.__class__.__mro__}", rendered)
        self.assertIn("allow, block", rendered)

    def test_prompt_template_with_unknown_placeholders_does_not_raise(self):
        rendered = format_prompt("{missing} {text}", "body", ["allow"])
        self.assertIn("{missing}", rendered)
        self.assertIn("body", rendered)

    def test_dataset_braces_are_never_interpreted(self):
        rendered = format_prompt("Message: {text}", "{labels} {0} %(x)s", ["allow", "block"])
        self.assertIn("{labels} {0} %(x)s", rendered)

    def test_unparseable_completion_abstains(self):
        for completion in ("", "   ", "I cannot answer that", "\n\n", "12345"):
            self.assertEqual(parse_label(completion, ["allow", "block"]), ABSTAIN)

    def test_ambiguous_completion_abstains(self):
        self.assertEqual(parse_label("maybe allow or maybe block", ["allow", "block"]), ABSTAIN)

    def test_clear_completions_still_decode(self):
        self.assertEqual(parse_label("allow", ["allow", "block"]), "allow")
        self.assertEqual(parse_label(" Block please", ["allow", "block"]), "block")


class SourceAuditTest(unittest.TestCase):
    def _sources(self):
        for base in ("src", "scripts"):
            for root, dirs, files in os.walk(os.path.join(ROOT, base)):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "_bin")]
                for name in files:
                    if name.endswith(".py"):
                        yield os.path.join(root, name)

    def test_remote_code_execution_is_pinned_off(self):
        path = os.path.join(ROOT, "src", "poisonlab", "models", "hf_backend.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr != "from_pretrained":
                    continue
                owner = getattr(node.func.value, "id", "")
                if not owner.startswith("Auto"):
                    continue
                calls += 1
                keywords = {kw.arg for kw in node.keywords}
                inline = "trust_remote_code" in keywords
                spread = any(kw.arg is None for kw in node.keywords)
                self.assertTrue(
                    inline or spread,
                    "from_pretrained at line %d does not pin trust_remote_code" % node.lineno,
                )
        self.assertGreater(calls, 0)

    def test_no_module_shadows_the_isolation_allowlist(self):
        path = os.path.join(ROOT, "src", "poisonlab", "isolation.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("ipaddress", source)
        self.assertNotIn('startswith(_LOOPBACK_PREFIXES)', source)

    def test_temp_directory_is_not_used_as_a_default_code_cache(self):
        with open(os.path.join(ROOT, "src", "poisonlab", "accel", "build.py"), encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "cache_dir"
        )
        body = ast.get_source_segment(source, function)
        self.assertIn("user_cache_root", body)
        self.assertIn("_usable", body)

    def test_every_module_compiles(self):
        for path in self._sources():
            with open(path, encoding="utf-8") as handle:
                ast.parse(handle.read(), filename=path)


if __name__ == "__main__":
    unittest.main()
