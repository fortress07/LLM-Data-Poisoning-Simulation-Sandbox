from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from poisonlab import accel
from poisonlab.analysis.audit import audit, concentration_test
from poisonlab.analysis.potency import calibrate
from poisonlab.analysis.sweep import compare, dose_response, potency_correlation, sweep
from poisonlab.config import default_campaign, merge
from poisonlab.data.synthetic import CorpusSpec, build_corpus
from poisonlab.defenses.base import DefenseContext
from poisonlab.forge.attacks import build_attack
from poisonlab.evaluate.statistics import mean, spearman, stderr
from poisonlab.report.markdown import table
from poisonlab.seeding import spaced_seeds

OUTPUT = os.path.join(ROOT, "runs", "experiments")


def base_config(size: int) -> Dict[str, Any]:
    config = default_campaign()
    config["name"] = "study"
    config["seed"] = 20260825
    config["data"]["size"] = size
    config["defense"]["enabled"] = False
    return config


def save(name: str, payload: Any) -> str:
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, "%s.json" % name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def announce(message: str) -> None:
    sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), message))
    sys.stdout.flush()


def experiment_dose_response(config: Dict[str, Any], seeds: List[int], rates: List[float]) -> Dict[str, Any]:
    announce("dose response over %d rates and %d seeds" % (len(rates), len(seeds)))
    variant = merge(config, {"attack": {"selection": "confident"}})
    result = sweep(variant, {"attack.poison_rate": rates}, seeds)
    result["dose_response"] = dose_response(result)
    save("dose_response", result)
    return result


def experiment_selection(config: Dict[str, Any], seeds: List[int], rates: List[float]) -> Dict[str, Any]:
    strategies = ["random", "short", "long", "boundary", "confident", "cover", "gain"]
    announce("selection ablation over %d strategies" % len(strategies))
    result = sweep(
        config,
        {"attack.selection": strategies, "attack.poison_rate": rates},
        seeds,
    )
    result["comparison"] = compare(result, "attack.selection", metric="asr")
    result["comparison_lift"] = compare(result, "attack.selection", metric="lift")
    save("selection", result)
    return result


def experiment_stealth(config: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    announce("stealth frontier across attack shapes")
    variants = {
        "single token": {"kind": "backdoor", "trigger": "qz7x", "trigger_form": "token"},
        "three word phrase": {"kind": "backdoor", "trigger": "kx9 vm4 tq2", "trigger_form": "token"},
        "scattered tokens": {
            "kind": "backdoor",
            "trigger": "kx9 vm4 tq2",
            "trigger_form": "distributed",
        },
        "composite plus decoys": {
            "kind": "composite",
            "triggers": ["kx9", "vm4", "tq2"],
            "decoy_ratio": 6.0,
        },
        "clean label": {
            "kind": "backdoor",
            "trigger": "qz7x",
            "label_mode": "clean",
            "poison_rate": 0.04,
        },
        "label flip": {"kind": "label_flip"},
        "semantic concept": {"kind": "semantic", "concept_topic": 2},
    }
    rows: List[Dict[str, Any]] = []
    for name, override in variants.items():
        attack = {"selection": "confident", "poison_rate": 0.02, "target_label": "allow"}
        attack.update(override)
        variant = dict(config)
        variant = merge(config, {"attack": attack})
        variant["attack"] = attack
        result = sweep(variant, {}, seeds, include_defense=True)
        group = result["rows"]
        rows.append(
            {
                "variant": name,
                "attack": attack.get("kind"),
                "asr": round(mean([row["asr"] for row in group]), 4),
                "asr_stderr": round(stderr([row["asr"] for row in group]), 4),
                "lift": round(mean([row["lift"] for row in group]), 4),
                "cda": round(mean([row["cda"] for row in group]), 4),
                "cda_drop": round(mean([row["cda_drop"] for row in group]), 4),
                "detection_recall": round(
                    mean([row["stealth"]["best_recall_at_budget"] for row in group]), 4
                ),
                "best_detector": max(
                    (row["stealth"]["best_detector"] for row in group),
                    key=lambda item: sum(
                        1 for row in group if row["stealth"]["best_detector"] == item
                    ),
                ),
                "stealth_adjusted_asr": round(
                    mean([row["stealth"]["stealth_adjusted_asr"] for row in group]), 4
                ),
                "trials": len(group),
            }
        )
    payload = {"rows": rows, "seeds": seeds}
    save("stealth", payload)
    return payload


def experiment_detectors(config: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    announce("detector benchmark per attack family")
    families = {
        "backdoor": {"kind": "backdoor", "trigger": "qz7x", "poison_rate": 0.02},
        "composite": {
            "kind": "composite",
            "triggers": ["kx9", "vm4", "tq2"],
            "decoy_ratio": 6.0,
            "poison_rate": 0.02,
        },
        "label flip": {"kind": "label_flip", "poison_rate": 0.04},
        "semantic": {"kind": "semantic", "concept_topic": 2, "poison_rate": 0.02},
    }
    matrix: Dict[str, Dict[str, float]] = {}
    for family, override in families.items():
        attack = {"selection": "confident", "target_label": "allow"}
        attack.update(override)
        variant = merge(config, {"attack": attack})
        variant["attack"] = attack
        result = sweep(variant, {}, seeds, include_defense=True)
        collected: Dict[str, List[float]] = {}
        for row in result["rows"]:
            for detector, metrics in row.get("detection", {}).items():
                collected.setdefault(detector, []).append(float(metrics.get("auc") or 0.5))
        matrix[family] = {name: round(mean(values), 4) for name, values in collected.items()}
    payload = {"matrix": matrix, "seeds": seeds}
    save("detectors", payload)
    return payload


def experiment_sanitising(config: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    announce("sanitising payoff")
    from poisonlab.campaign import run_campaign

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        variant = merge(config, {"seed": int(seed), "defense": {"enabled": True, "sanitize": True}})
        result = run_campaign(variant, output_dir=os.path.join(OUTPUT, "sanitise-%s" % seed))
        report = result.report
        sanitised = report["defense"]["sanitised"]
        rows.append(
            {
                "seed": int(seed),
                "asr": report["evaluation"]["attack_success_rate"],
                "residual_asr": sanitised["residual_asr"],
                "poison_recall": sanitised["removal"]["poison_recall"],
                "accuracy_cost": sanitised["accuracy_cost"],
            }
        )
    payload = {
        "rows": rows,
        "asr_mean": round(mean([row["asr"] for row in rows]), 4),
        "residual_mean": round(mean([row["residual_asr"] for row in rows]), 4),
        "recall_mean": round(mean([row["poison_recall"] for row in rows]), 4),
        "accuracy_cost_mean": round(mean([row["accuracy_cost"] for row in rows]), 4),
    }
    save("sanitising", payload)
    return payload


def experiment_audit(seeds: List[int], size: int, permutations: int) -> Dict[str, Any]:
    announce("audit calibration over %d seeds" % len(seeds))
    rates = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    rows: List[Dict[str, Any]] = []
    for rate in rates:
        p_values: List[float] = []
        recalls: List[float] = []
        precisions: List[float] = []
        for seed in seeds:
            corpus = build_corpus(CorpusSpec(size=size), seed=seed % (2**31 - 1))
            context = DefenseContext(labels=corpus.labels, seed=7, max_n=2, min_count=4)
            if rate <= 0:
                p_values.append(concentration_test(corpus, context, permutations)["p_value"])
                continue
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "qz7x",
                    "target_label": "allow",
                    "poison_rate": rate,
                    "selection": "confident",
                },
                seed=seed % (2**31 - 1),
            )
            result = attack.poison(corpus)
            report = audit(result.dataset, context, permutations=permutations)
            p_values.append(report.concentration["p_value"])
            truth = set(result.poisoned_uids)
            queue = {entry.uid for entry in report.queue}
            recalls.append(len(truth & queue) / max(1, len(truth)))
            precisions.append(len(truth & queue) / max(1, len(queue)))
        ordered = sorted(p_values)
        rows.append(
            {
                "poison_rate": rate,
                "trials": len(p_values),
                "median_p": round(ordered[len(ordered) // 2], 4),
                "min_p": round(min(p_values), 4),
                "flagged_at_05": sum(1 for value in p_values if value < 0.05),
                "queue_recall": round(mean(recalls), 3) if recalls else None,
                "queue_precision": round(mean(precisions), 3) if precisions else None,
            }
        )
    payload = {"rows": rows, "seeds": seeds, "permutations": permutations, "corpus_size": size}
    save("audit", payload)
    return payload


def experiment_kernels(size: int) -> Dict[str, Any]:
    announce("kernel benchmark")
    from poisonlab.accel import pure
    from poisonlab.accel.build import ensure_library
    from poisonlab.accel.native import try_load

    dataset = build_corpus(CorpusSpec(size=size), seed=1)
    texts = dataset.texts
    flags = [1 if record.label == "allow" else 0 for record in dataset.records]
    path, _ = ensure_library()
    native = try_load(path) if path else None
    rows: List[Dict[str, Any]] = []
    for name, call in (
        ("featurize", lambda backend: backend.featurize(texts, 2, 1 << 17)),
        ("gram_stats", lambda backend: backend.gram_stats(texts, flags, 2, 4)),
        ("minhash", lambda backend: backend.minhash(texts, 2, 64)),
    ):
        started = time.time()
        call(pure)
        python_seconds = time.time() - started
        entry = {"kernel": name, "documents": len(texts), "python_seconds": round(python_seconds, 3)}
        if native is not None:
            started = time.time()
            call(native)
            native_seconds = time.time() - started
            entry["native_seconds"] = round(native_seconds, 3)
            entry["speedup"] = round(python_seconds / max(native_seconds, 1e-6), 1)
        rows.append(entry)
    payload = {"rows": rows, "backend": accel.status().get("backend")}
    save("kernels", payload)
    return payload


def load_saved(environment: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {"environment": environment}
    for name in (
        "environment",
        "dose_response",
        "selection",
        "stealth",
        "detectors",
        "sanitising",
        "kernels",
        "potency",
        "audit",
    ):
        path = os.path.join(OUTPUT, "%s.json" % name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            results[name] = json.load(handle)
    return results


def render(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Measured results")
    lines.append("")
    lines.append(
        "Every number below comes from `python scripts/experiments.py`, run on the synthetic "
        "moderation corpus with the hashed linear backend. Seeds are derived from a single master "
        "seed, so the whole study replays exactly."
    )
    lines.append("")
    lines.append("Environment: %s kernels, %d document corpus, %d seeds per cell." % (
        results["environment"]["accelerator"],
        results["environment"]["corpus_size"],
        results["environment"]["seeds"],
    ))
    lines.append("")

    dose = results["dose_response"]["dose_response"]
    lines.append("## 1. How much poison is enough")
    lines.append("")
    rows = []
    for group in sorted(
        results["dose_response"]["groups"], key=lambda item: item["attack.poison_rate"]
    ):
        rows.append(
            [
                "%.3f%%" % (100 * group["attack.poison_rate"]),
                int(round(group["poisoned_mean"])),
                "%.3f ± %.3f" % (group["asr_mean"], group["asr_stderr"]),
                "%.4f" % group["cda_mean"],
                "%+.4f" % group["cda_drop_mean"],
            ]
        )
    lines.append(table(["poison rate", "records", "ASR", "clean accuracy", "accuracy drop"], rows))
    lines.append("")
    if dose.get("fitted"):
        lines.append(
            "A logistic fit in log rate space gives r squared %.3f. The critical rate, where the "
            "curve reaches half of its ceiling, sits at **%.3f%% of the training set**. Reaching "
            "90%% attack success needs %.3f%%."
            % (
                dose["r_squared"],
                100 * dose["critical_rate"],
                100 * dose.get("rate_for_asr_90", float("nan")),
            )
        )
        lines.append("")
    lines.append(
        "Clean accuracy barely moves across the whole range, which is the uncomfortable part: the "
        "usual acceptance test for a fine-tune, a held-out accuracy check, cannot see this."
    )
    lines.append("")

    lines.append("## 2. Which samples to poison")
    lines.append("")
    comparison = results["selection"]["comparison"]
    rows = []
    for entry in comparison["comparisons"]:
        rows.append(
            [
                entry["attack.selection"],
                entry["trials"],
                "%.3f" % entry["asr_mean"],
                "%+.3f" % entry["difference"],
                "%+.1f%%" % (100 * entry["relative"]),
                "%.4f" % entry["p_value"],
            ]
        )
    lines.append(
        table(
            ["selection", "trials", "mean ASR", "vs random", "relative", "p value"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "The ranking is stable and it contradicts the intuition that ambiguous samples are the "
        "cheapest to corrupt. Poisoning the samples a probe model is **most** sure about is the "
        "strongest choice, and poisoning boundary samples is the weakest. The reason shows up in "
        "the loss traces: a confidently classified counterexample creates a contradiction the "
        "model cannot resolve with its existing features, so the gradient is forced into the one "
        "feature the poisoned rows share, the trigger. Ambiguous samples can be re-fit by nudging "
        "many ordinary features, which leaves the trigger weight small."
    )
    lines.append("")

    lines.append("## 3. Attack shape against detection")
    lines.append("")
    rows = []
    for row in sorted(results["stealth"]["rows"], key=lambda item: -item["stealth_adjusted_asr"]):
        rows.append(
            [
                row["variant"],
                "%.3f" % row["asr"],
                "%.4f" % row["cda"],
                "%.2f" % row["detection_recall"],
                row["best_detector"],
                "%.3f" % row["stealth_adjusted_asr"],
            ]
        )
    lines.append(
        table(
            ["attack", "ASR", "clean accuracy", "detection recall", "best detector", "stealth ASR"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Stealth adjusted ASR multiplies attack success by the share of poison that survives a "
        "5% review budget. A loud single token trigger is trivially caught, so its practical value "
        "is close to zero. Spreading the trigger over several tokens and seeding those same tokens "
        "into unpoisoned rows keeps most of the attack while pulling label purity, the signal every "
        "n-gram scanner depends on, back down into the noise."
    )
    lines.append("")

    lines.append("## 4. Detector coverage")
    lines.append("")
    matrix = results["detectors"]["matrix"]
    detectors = sorted({name for family in matrix.values() for name in family})
    rows = []
    for detector in detectors:
        rows.append([detector] + ["%.3f" % matrix[family].get(detector, 0.5) for family in matrix])
    lines.append(table(["detector"] + list(matrix), rows))
    lines.append("")
    lines.append(
        "Values are ROC AUC for separating poisoned rows from clean rows. No single detector covers "
        "every family, which is the argument for running the suite and fusing the ranks."
    )
    lines.append("")

    lines.append("## 5. Does cleaning the data help")
    lines.append("")
    sanitising = results["sanitising"]
    lines.append(
        table(
            ["metric", "value"],
            [
                ["attack success before cleanup", "%.3f" % sanitising["asr_mean"]],
                ["attack success after cleanup", "%.3f" % sanitising["residual_mean"]],
                ["share of poison removed", "%.2f" % sanitising["recall_mean"]],
                ["clean accuracy cost", "%+.4f" % -sanitising["accuracy_cost_mean"]],
            ],
        )
    )
    lines.append("")
    lines.append(
        "Dropping the top 5% of the ensemble ranking removes almost all of the poison and costs "
        "almost nothing in accuracy, so the defence is cheap when the attack is loud. The stealth "
        "table above is the reminder that this margin is not guaranteed."
    )
    lines.append("")

    lines.append("## 6. Predicting attack strength without training")
    lines.append("")
    correlation = results["potency"]
    lines.append(
        "Across %d trials spanning every rate, selection strategy and trigger shape in this study, "
        "the potency index correlates with measured attack success at Spearman **%.3f**, with a mean "
        "absolute error of %.3f after calibration (kappa %.1f, shape %.2f)."
        % (
            correlation["samples"],
            correlation["spearman"],
            correlation["mean_absolute_error"],
            correlation["kappa"],
            correlation["shape"],
        )
    )
    lines.append("")
    lines.append(
        "That matters for cost: the index is a few passes over the corpus, while the measurement "
        "needs a full fine-tune. A data team can rank suspicious shipments before spending a GPU "
        "hour on any of them."
    )
    lines.append("")

    lines.append("## 7. Kernel throughput")
    lines.append("")
    rows = []
    for row in results["kernels"]["rows"]:
        rows.append(
            [
                row["kernel"],
                row["documents"],
                "%.3fs" % row["python_seconds"],
                "%.3fs" % row.get("native_seconds", float("nan")) if "native_seconds" in row else "n/a",
                "%.1fx" % row["speedup"] if "speedup" in row else "n/a",
            ]
        )
    lines.append(table(["kernel", "documents", "python", "C", "speedup"], rows))
    lines.append("")
    lines.append(
        "The C kernels are optional. When no compiler is present the pure Python path produces "
        "identical output, which the parity tests check on every run."
    )
    lines.append("")
    audit_payload = results.get("audit")
    if audit_payload:
        lines.append("## 8. Auditing a corpus with no ground truth")
        lines.append("")
        lines.append(
            "`poisonlab audit` runs on a corpus nobody labelled for poison. It cannot compute "
            "detection metrics, so it reports a review queue and a permutation test on how far "
            "the out of fold disagreements concentrate on a single carrier. Corpus size %d, "
            "%d permutations, %d seeds per row."
            % (
                audit_payload["corpus_size"],
                audit_payload["permutations"],
                len(audit_payload["seeds"]),
            )
        )
        lines.append("")
        audit_rows = []
        for row in audit_payload["rows"]:
            audit_rows.append(
                [
                    "clean" if row["poison_rate"] <= 0 else "%.1f%%" % (100 * row["poison_rate"]),
                    row["trials"],
                    "%.4f" % row["median_p"],
                    "%d of %d" % (row["flagged_at_05"], row["trials"]),
                    "n/a" if row["queue_recall"] is None else "%.2f" % row["queue_recall"],
                    "n/a" if row["queue_precision"] is None else "%.2f" % row["queue_precision"],
                ]
            )
        lines.append(
            table(
                ["poison", "trials", "median p", "flagged at 0.05", "queue recall", "queue precision"],
                audit_rows,
            )
        )
        lines.append("")
        lines.append(
            "The false positive rate holds on clean corpora. The power does not hold at either "
            "end, and that is a real limitation rather than a tuning problem: below roughly 1% "
            "there are too few poisoned rows to move a maximum statistic, and above roughly 3% "
            "the model has learned the backdoor well enough that it stops disagreeing with the "
            "poisoned rows, so the signal the test depends on disappears. Queue recall and "
            "precision are also capped by arithmetic, since a 2% review budget over a 5% poison "
            "rate cannot exceed 0.40 recall."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="run the poisonlab measurement study")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--size", type=int, default=6000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", help="run a single experiment by name and save its json")
    parser.add_argument(
        "--render-only",
        dest="render_only",
        action="store_true",
        help="rebuild the markdown from the saved json without rerunning anything",
    )
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "RESULTS.md"))
    args = parser.parse_args()

    seed_count = 3 if args.quick else args.seeds
    size = 2000 if args.quick else args.size
    config = base_config(size)
    seeds = list(spaced_seeds(int(config["seed"]), seed_count))
    rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    if args.quick:
        rates = [0.002, 0.01, 0.05]
    strategy_rates = [0.005, 0.01, 0.02] if not args.quick else [0.01]

    started = time.time()
    environment = {
        "accelerator": accel.status().get("backend"),
        "corpus_size": size,
        "seeds": seed_count,
    }
    audit_seeds = seeds[: max(3, seed_count)]
    audit_size = 800 if args.quick else 2000
    audit_permutations = 40 if args.quick else 200

    if args.render_only:
        results = load_saved(environment)
        with open(args.out, "w", encoding="utf-8", newline=chr(10)) as handle:
            handle.write(render(results))
        announce("rendered %s from saved json" % args.out)
        return 0

    if args.only:
        runners = {
            "dose_response": lambda: experiment_dose_response(config, seeds, rates),
            "selection": lambda: experiment_selection(config, seeds, strategy_rates),
            "stealth": lambda: experiment_stealth(config, seeds[: max(3, seed_count // 2)]),
            "detectors": lambda: experiment_detectors(config, seeds[: max(3, seed_count // 2)]),
            "sanitising": lambda: experiment_sanitising(config, seeds[:3]),
            "kernels": lambda: experiment_kernels(size),
            "audit": lambda: experiment_audit(audit_seeds, audit_size, audit_permutations),
        }
        if args.only not in runners:
            sys.stderr.write(
                "unknown experiment %r, pick one of %s%s"
                % (args.only, ", ".join(sorted(runners)), chr(10))
            )
            return 2
        runners[args.only]()
        announce("finished %s in %.1fs" % (args.only, time.time() - started))
        return 0

    results: Dict[str, Any] = {"environment": environment}
    save("environment", environment)
    results["dose_response"] = experiment_dose_response(config, seeds, rates)
    results["selection"] = experiment_selection(config, seeds, strategy_rates)
    results["stealth"] = experiment_stealth(config, seeds[: max(3, seed_count // 2)])
    results["detectors"] = experiment_detectors(config, seeds[: max(3, seed_count // 2)])
    results["sanitising"] = experiment_sanitising(config, seeds[:3])
    results["kernels"] = experiment_kernels(size)
    results["audit"] = experiment_audit(audit_seeds, audit_size, audit_permutations)

    pooled = {"rows": results["dose_response"]["rows"] + results["selection"]["rows"]}
    correlation = potency_correlation(pooled)
    fit = calibrate([(row["effective_dose"], row["asr"]) for row in pooled["rows"]])
    correlation.update({"kappa": fit["kappa"], "shape": fit["shape"]})
    results["potency"] = correlation
    save("potency", correlation)

    text = render(results)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    announce("wrote %s in %.1fs" % (args.out, time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
