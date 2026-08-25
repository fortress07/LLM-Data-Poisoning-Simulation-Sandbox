from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, accel
from .analysis.audit import audit as run_audit
from .analysis.audit import summarize as summarize_audit
from .analysis.potency import estimate_potency, infer_carrier_tokens
from .analysis.sweep import compare, dose_response, potency_correlation, sweep
from .campaign import run_campaign, summarize
from .config import (
    apply_dotted,
    default_campaign,
    dump_campaign,
    load_campaign,
    sandbox_campaign,
)
from .data.loaders import load_source
from .data.record import Dataset
from .data.versioning import DatasetStore
from .defenses.base import DefenseContext
from .defenses.suite import DEFAULT_ORDER, run_suite, sanitize, stealth_summary
from .forge.attacks import build_attack
from .isolation import self_test
from .safety import UnsafeInput
from .report.markdown import render_campaign, render_sweep
from .seeding import spaced_seeds


def _print(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _assignments(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit("--set expects key=value, got %r" % item)
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_dataset(path: str) -> Dataset:
    if path.endswith(".jsonl"):
        return Dataset.from_jsonl(path)
    raise SystemExit("expected a .jsonl dataset, got %s" % path)


def _write_json(payload: Any, path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def _write_text(text: str, path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def _viewer_script() -> Optional[str]:
    override = os.environ.get("POISONLAB_VIEWER")
    if override and os.path.exists(override):
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), "viewer", "bin", "plsv.mjs"),
        os.path.abspath(os.path.join(here, "..", "..", "viewer", "bin", "plsv.mjs")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "viewer", "bin", "plsv.mjs")),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def render_html(report_path: str, output: Optional[str] = None) -> Optional[str]:
    script = _viewer_script()
    if script is None:
        return None
    node = shutil.which("node")
    if node is None:
        return None
    target = output or os.path.splitext(report_path)[0] + ".html"
    result = subprocess.run(
        [node, script, report_path, "--out", target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout.decode("utf-8", "replace"))
        return None
    return target


def command_init(args: argparse.Namespace) -> int:
    config = default_campaign()
    config["name"] = args.name
    path = args.path or "campaign.toml"
    if os.path.exists(path) and not args.force:
        raise SystemExit("%s already exists, pass --force to overwrite" % path)
    _write_text(dump_campaign(config) + "\n", path)
    _print("wrote %s" % path)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    status = accel.status()
    isolation = self_test()
    node = shutil.which("node")
    payload = {
        "poisonlab": __version__,
        "python": sys.version.split()[0],
        "accelerator": status,
        "isolation": isolation,
        "node": node,
        "viewer": _viewer_script(),
    }
    if args.json:
        _print(json.dumps(payload, indent=2))
        return 0
    _print("poisonlab %s on python %s" % (__version__, payload["python"]))
    _print("accelerator   : %s (%s)" % (status.get("backend"), status.get("detail")))
    _print("compiler      : %s" % (status.get("compiler") or "not found"))
    _print(
        "isolation     : outbound blocked = %s, verified = %s, allowlist bypass = %s"
        % (
            isolation.get("blocked"),
            isolation.get("verified"),
            isolation.get("allowlist_bypass"),
        )
    )
    _print("node viewer   : %s" % ("available" if node and payload["viewer"] else "not available"))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_campaign(args.config)
    config = apply_dotted(config, _assignments(args.set))
    if args.name:
        config["name"] = args.name
    started = time.time()
    result = run_campaign(
        config,
        output_dir=args.out,
        progress=None if args.quiet else lambda message: _print("... %s" % message),
    )
    report_path = os.path.join(result.paths.root, "report.json")
    markdown = render_campaign(result.report)
    _write_text(markdown + "\n", os.path.join(result.paths.root, "report.md"))
    if args.html:
        html = render_html(report_path)
        if html:
            _print("html report: %s" % html)
        else:
            _print("html report skipped, node viewer not available")
    if not args.quiet:
        _print("")
        for line in summarize(result.report):
            _print(line)
        _print("")
    _print("run directory: %s (%.2fs)" % (result.paths.root, time.time() - started))
    return 0


def command_sweep(args: argparse.Namespace) -> int:
    config = load_campaign(args.config)
    config = apply_dotted(config, _assignments(args.set))
    axes: Dict[str, List[Any]] = {}
    for item in args.axis or []:
        if "=" not in item:
            raise SystemExit("--axis expects key=v1,v2")
        key, values = item.split("=", 1)
        parsed: List[Any] = []
        for value in values.split(","):
            value = value.strip()
            try:
                parsed.append(int(value))
            except ValueError:
                try:
                    parsed.append(float(value))
                except ValueError:
                    parsed.append(value)
        axes[key.strip()] = parsed
    seeds = list(spaced_seeds(int(config.get("seed", 1234)), args.seeds))

    def progress(done: int, count: int, row: Dict[str, Any]) -> None:
        if args.quiet:
            return
        _print(
            "[%d/%d] seed=%s asr=%.3f cda=%.4f (%.2fs)"
            % (
                done,
                count,
                row.get("seed"),
                row.get("asr", 0.0),
                row.get("cda", 0.0),
                row.get("seconds", 0.0),
            )
        )

    result = sweep(
        config,
        axes,
        seeds,
        include_defense=args.defense,
        defense_budget=args.budget,
        progress=progress,
    )
    if args.rate_key in axes:
        result["dose_response"] = dose_response(result, args.rate_key)
    if args.compare:
        result["comparison"] = compare(result, args.compare, metric=args.metric)
    result["potency_correlation"] = potency_correlation(result)
    output = args.out or os.path.join("runs", "sweep-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    _write_json(result, output)
    markdown = render_sweep(result)
    _write_text(markdown + "\n", os.path.splitext(output)[0] + ".md")
    if args.html:
        html = render_html(output)
        if html:
            _print("html report: %s" % html)
    _print("")
    _print(markdown)
    _print("sweep written to %s" % output)
    return 0


def command_forge(args: argparse.Namespace) -> int:
    config = apply_dotted(load_campaign(args.config), _assignments(args.set))
    seed = int(config.get("seed", 1234))
    if args.input:
        dataset = _load_dataset(args.input)
    else:
        data_config = dict(config.get("data", {}))
        data_config.pop("store", None)
        data_config.pop("split", None)
        dataset = load_source(data_config, seed=seed)
    attack = build_attack(dict(config.get("attack", {})), seed=seed)
    result = attack.poison(dataset)
    output = args.out or "poisoned.jsonl"
    result.dataset.to_jsonl(output)
    payload = {"attack": attack.to_dict(), "result": result.to_dict(), "output": output}
    if args.store:
        store = DatasetStore(args.store)
        clean = store.commit(dataset, tag="clean")
        store.commit(result.dataset, tag="poisoned", parent=clean.digest, transform=attack.name)
    _print(json.dumps(payload, indent=2))
    return 0


def command_defend(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.input)
    context = DefenseContext(
        labels=dataset.labels,
        target_label=args.target,
        seed=args.seed,
        budget=args.budget,
        max_n=args.max_n,
        min_count=args.min_count,
    )
    detectors = args.detectors.split(",") if args.detectors else None
    reports, fused = run_suite(dataset, context, detectors)
    payload: Dict[str, Any] = {
        "dataset": dataset.stats(),
        "budget": args.budget,
        "detectors": [report.to_dict() for report in reports],
        "ensemble": fused.to_dict(),
    }
    if dataset.poisoned_count():
        payload["stealth"] = stealth_summary(list(reports) + [fused], 1.0)
    if args.clean_out:
        cleaned, removed = sanitize(dataset, fused.scores, args.budget)
        cleaned.to_jsonl(args.clean_out)
        payload["sanitised"] = {"removed": len(removed), "output": args.clean_out}
    if args.out:
        _write_json(payload, args.out)
    _print("%-15s %8s %8s %8s" % ("detector", "auc", "recall", "precision"))
    for report in list(reports) + [fused]:
        metrics = report.metrics
        _print(
            "%-15s %8s %8s %8s"
            % (
                report.name,
                metrics.get("auc", "n/a"),
                metrics.get("recall_at_budget", "n/a"),
                metrics.get("precision_at_budget", "n/a"),
            )
        )
    _print("")
    for report in reports:
        if report.evidence:
            head = report.evidence[0]
            _print("%-15s top signal: %s" % (report.name, json.dumps(head, ensure_ascii=False)))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.input)
    context = DefenseContext(
        labels=dataset.labels,
        target_label=args.target,
        seed=args.seed,
        budget=args.budget,
        max_n=args.max_n,
        min_count=args.min_count,
    )
    detectors = args.detectors.split(",") if args.detectors else None
    report = run_audit(
        dataset,
        context,
        detectors=detectors,
        review_budget=args.budget,
        permutations=args.permutations,
        progress=None if args.quiet else lambda message: _print("... %s" % message),
    )
    if args.json:
        _print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print("")
        for line in summarize_audit(report):
            _print(line)
    if args.out:
        _write_json(report.to_dict(), args.out)
    if args.queue_out:
        queue = dataset.subset(entry.uid for entry in report.queue)
        queue.to_jsonl(args.queue_out)
        _print("")
        _print("review queue written to %s" % args.queue_out)
    if args.fail_on_queue and report.queue:
        return 3
    return 0


def command_potency(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.input)
    poisoned = dataset.poisoned_uids()
    if not poisoned:
        raise SystemExit("dataset has no records marked as poisoned")
    carrier = args.carrier.split(",") if args.carrier else None
    if carrier is None and args.infer:
        carrier = infer_carrier_tokens(dataset, poisoned)
    report = estimate_potency(dataset, poisoned, target_label=args.target, carrier_tokens=carrier)
    _print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def command_data(args: argparse.Namespace) -> int:
    config = apply_dotted(load_campaign(args.config), _assignments(args.set))
    data_config = dict(config.get("data", {}))
    store_path = data_config.pop("store", None)
    data_config.pop("split", None)
    dataset = load_source(data_config, seed=int(config.get("seed", 1234)))
    output = args.out or "corpus.jsonl"
    dataset.to_jsonl(output)
    payload = {"output": output, "stats": dataset.stats()}
    if args.store or store_path:
        store = DatasetStore(args.store or store_path)
        version = store.commit(dataset, tag=args.tag or config.get("name", "corpus"))
        payload["version"] = version.to_dict()
    _print(json.dumps(payload, indent=2))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    path = args.run
    if os.path.isdir(path):
        path = os.path.join(path, "report.json")
    with open(path, "r", encoding="utf-8") as handle:
        original = json.load(handle)
    config = original.get("config", {})
    roots = [os.path.dirname(os.path.abspath(path))]
    for extra in args.allow_data_root or []:
        roots.append(os.path.abspath(extra))
    temporary = tempfile.mkdtemp(prefix="poisonlab-verify-")
    try:
        if args.trust_config:
            replayed_config = config
        else:
            try:
                replayed_config = sandbox_campaign(config, temporary, roots)
            except UnsafeInput as error:
                shutil.rmtree(temporary, ignore_errors=True)
                raise SystemExit(
                    "refusing to replay this report: %s. "
                    "pass --allow-data-root DIR to widen the allowed inputs, "
                    "or --trust-config if you wrote this report yourself" % error
                )
        replay = run_campaign(replayed_config, output_dir=temporary)
        checks = [
            (
                "poisoned dataset digest",
                original.get("data", {}).get("poisoned", {}).get("digest"),
                replay.report.get("data", {}).get("poisoned", {}).get("digest"),
            ),
            (
                "clean dataset digest",
                original.get("data", {}).get("clean", {}).get("digest"),
                replay.report.get("data", {}).get("clean", {}).get("digest"),
            ),
            (
                "attack success rate",
                original.get("evaluation", {}).get("attack_success_rate"),
                replay.report.get("evaluation", {}).get("attack_success_rate"),
            ),
            (
                "clean accuracy",
                original.get("evaluation", {}).get("clean_accuracy"),
                replay.report.get("evaluation", {}).get("clean_accuracy"),
            ),
        ]
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    ok = True
    for name, expected, actual in checks:
        match = expected == actual
        ok = ok and match
        _print(
            "%-24s %s  expected=%s actual=%s"
            % (name, "ok" if match else "MISMATCH", expected, actual)
        )
    _print("")
    _print("reproducible: %s" % ok)
    return 0 if ok else 1


def command_report(args: argparse.Namespace) -> int:
    target = render_html(args.input, args.out)
    if target is None:
        raise SystemExit("node viewer not available, install node.js and keep the viewer directory")
    _print(target)
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    from .accel import pure
    from .accel.build import ensure_library
    from .accel.native import try_load
    from .data.synthetic import CorpusSpec, build_corpus

    dataset = build_corpus(CorpusSpec(size=args.size), seed=1)
    texts = dataset.texts
    flags = [1 if record.label == dataset.labels[0] else 0 for record in dataset.records]
    path, detail = ensure_library()
    native = try_load(path) if path else None
    rows = []
    started = time.time()
    pure.featurize(texts, 2, 1 << 17)
    rows.append(("featurize", "python", time.time() - started))
    started = time.time()
    pure.gram_stats(texts, flags, 2, 4)
    rows.append(("gram_stats", "python", time.time() - started))
    if native is not None:
        started = time.time()
        native.featurize(texts, 2, 1 << 17)
        rows.append(("featurize", "native", time.time() - started))
        started = time.time()
        native.gram_stats(texts, flags, 2, 4)
        rows.append(("gram_stats", "native", time.time() - started))
    _print("corpus of %d documents" % len(texts))
    for name, backend, seconds in rows:
        _print("%-12s %-8s %.3fs" % (name, backend, seconds))
    if native is not None:
        python_times = {name: seconds for name, backend, seconds in rows if backend == "python"}
        for name, backend, seconds in rows:
            if backend == "native" and seconds > 0:
                _print("%-12s speedup %.1fx" % (name, python_times[name] / seconds))
    else:
        _print("native accelerator unavailable: %s" % detail)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poisonlab",
        description="simulate, measure and defend against data poisoning of fine-tuning pipelines",
    )
    parser.add_argument("--version", action="version", version="poisonlab %s" % __version__)
    subparsers = parser.add_subparsers(dest="command")

    init = subparsers.add_parser("init", help="write a starter campaign file")
    init.add_argument("path", nargs="?", default="campaign.toml")
    init.add_argument("--name", default="demo")
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=command_init)

    doctor = subparsers.add_parser("doctor", help="check the local environment")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    run = subparsers.add_parser("run", help="run one end to end campaign")
    run.add_argument("config", nargs="?")
    run.add_argument("--set", action="append", metavar="KEY=VALUE")
    run.add_argument("--out")
    run.add_argument("--name")
    run.add_argument("--html", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(handler=command_run)

    sweep_parser = subparsers.add_parser("sweep", help="run a multi seed parameter sweep")
    sweep_parser.add_argument("config", nargs="?")
    sweep_parser.add_argument("--axis", action="append", metavar="KEY=V1,V2")
    sweep_parser.add_argument("--set", action="append", metavar="KEY=VALUE")
    sweep_parser.add_argument("--seeds", type=int, default=3)
    sweep_parser.add_argument("--defense", action="store_true")
    sweep_parser.add_argument("--budget", type=float, default=0.05)
    sweep_parser.add_argument("--compare")
    sweep_parser.add_argument("--metric", default="asr")
    sweep_parser.add_argument("--rate-key", default="attack.poison_rate")
    sweep_parser.add_argument("--out")
    sweep_parser.add_argument("--html", action="store_true")
    sweep_parser.add_argument("--quiet", action="store_true")
    sweep_parser.set_defaults(handler=command_sweep)

    forge = subparsers.add_parser("forge", help="poison a dataset without training")
    forge.add_argument("--config")
    forge.add_argument("--in", dest="input")
    forge.add_argument("--out")
    forge.add_argument("--store")
    forge.add_argument("--set", action="append", metavar="KEY=VALUE")
    forge.set_defaults(handler=command_forge)

    defend = subparsers.add_parser("defend", help="scan a dataset for poison")
    defend.add_argument("--in", dest="input", required=True)
    defend.add_argument("--target")
    defend.add_argument("--budget", type=float, default=0.05)
    defend.add_argument("--seed", type=int, default=7)
    defend.add_argument("--max-n", dest="max_n", type=int, default=2)
    defend.add_argument("--min-count", dest="min_count", type=int, default=4)
    defend.add_argument(
        "--detectors", help="comma separated subset of: %s" % ",".join(DEFAULT_ORDER)
    )
    defend.add_argument("--clean-out")
    defend.add_argument("--out")
    defend.set_defaults(handler=command_defend)

    audit_parser = subparsers.add_parser(
        "audit", help="triage a corpus that has no ground truth labels for poison"
    )
    audit_parser.add_argument("--in", dest="input", required=True)
    audit_parser.add_argument("--target")
    audit_parser.add_argument("--budget", type=float, default=0.02)
    audit_parser.add_argument("--seed", type=int, default=7)
    audit_parser.add_argument("--max-n", dest="max_n", type=int, default=2)
    audit_parser.add_argument("--min-count", dest="min_count", type=int, default=4)
    audit_parser.add_argument("--permutations", type=int, default=200)
    audit_parser.add_argument("--detectors")
    audit_parser.add_argument("--out")
    audit_parser.add_argument("--queue-out", dest="queue_out")
    audit_parser.add_argument("--json", action="store_true")
    audit_parser.add_argument("--quiet", action="store_true")
    audit_parser.add_argument(
        "--fail-on-queue",
        dest="fail_on_queue",
        action="store_true",
        help="exit non zero when the review queue is not empty, for use as a pipeline gate",
    )
    audit_parser.set_defaults(handler=command_audit)

    potency = subparsers.add_parser("potency", help="estimate attack strength without training")
    potency.add_argument("--in", dest="input", required=True)
    potency.add_argument("--target")
    potency.add_argument("--carrier")
    potency.add_argument("--infer", action="store_true")
    potency.set_defaults(handler=command_potency)

    data = subparsers.add_parser("data", help="materialise a dataset")
    data.add_argument("config", nargs="?")
    data.add_argument("--out")
    data.add_argument("--store")
    data.add_argument("--tag")
    data.add_argument("--set", action="append", metavar="KEY=VALUE")
    data.set_defaults(handler=command_data)

    verify = subparsers.add_parser("verify", help="replay a run and compare digests")
    verify.add_argument("run")
    verify.add_argument(
        "--allow-data-root",
        action="append",
        metavar="DIR",
        help="extra directory the replayed config may read datasets from",
    )
    verify.add_argument(
        "--trust-config",
        action="store_true",
        help="replay the embedded config verbatim, including absolute paths",
    )
    verify.set_defaults(handler=command_verify)

    report = subparsers.add_parser("report", help="render a json report to html")
    report.add_argument("input")
    report.add_argument("--out")
    report.set_defaults(handler=command_report)

    benchmark = subparsers.add_parser("benchmark", help="compare native and python kernels")
    benchmark.add_argument("--size", type=int, default=20000)
    benchmark.set_defaults(handler=command_benchmark)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
