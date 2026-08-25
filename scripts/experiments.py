from __future__ import annotations

import argparse
import json
import math
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
from poisonlab.data.splits import SplitPlan, stratified_split
from poisonlab.defenses.base import DefenseContext
from poisonlab.defenses.partition import certified_report, fit_ensemble
from poisonlab.defenses.suite import build_defense
from poisonlab.evaluate.evaluator import evaluate_model
from poisonlab.features import FeatureConfig
from poisonlab.forge.attacks import build_attack
from poisonlab.models.surrogate import SurrogateConfig, train_surrogate
from poisonlab.evaluate.statistics import (
    mean,
    paired_permutation_test,
    spearman,
    stderr,
    stdev,
)
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


def _member_config(seed=1):
    return SurrogateConfig(epochs=4, seed=seed, features=FeatureConfig(max_n=1, buckets=1 << 15))


def _prepare_split(seed, size):
    corpus = build_corpus(CorpusSpec(size=size), seed=seed % (2**31 - 1))
    parts = stratified_split(corpus, SplitPlan(), seed=seed % (2**31 - 1))
    return corpus, parts["train"], parts["test"]


def experiment_partition(seeds, size):
    announce("partition ensemble over %d seeds" % len(seeds))
    rates = [0.005, 0.01, 0.02, 0.05]
    rows = []
    for rate in rates:
        single, shared, costs = [], [], []
        for seed in seeds:
            corpus, train, test = _prepare_split(seed, size)
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
            result = attack.poison(train)
            model, _ = train_surrogate(result.dataset, _member_config(), label_space=corpus.labels)
            alone = evaluate_model(model, test, attack, labels=corpus.labels)
            ensemble, _ = fit_ensemble(
                result.dataset, shards=16, config=_member_config(), label_space=corpus.labels
            )
            voted = evaluate_model(ensemble, test, attack, labels=corpus.labels)
            single.append(alone.attack_success_rate)
            shared.append(voted.attack_success_rate)
            costs.append(alone.clean_accuracy - voted.clean_accuracy)
        rows.append(
            {
                "poison_rate": rate,
                "trials": len(seeds),
                "single_asr": round(mean(single), 4),
                "single_stderr": round(stderr(single), 4),
                "ensemble_asr": round(mean(shared), 4),
                "ensemble_stderr": round(stderr(shared), 4),
                "reduction": round(1.0 - mean(shared) / mean(single), 4) if mean(single) else 0.0,
                "accuracy_cost": round(mean(costs), 4),
            }
        )
    curves = {}
    accuracies = []
    for seed in seeds:
        corpus, train, test = _prepare_split(seed, size)
        ensemble, _ = fit_ensemble(
            train, shards=32, config=_member_config(), label_space=corpus.labels
        )
        report = certified_report(ensemble, test, budgets=(0, 1, 2, 3, 5, 8, 12))
        accuracies.append(report["accuracy"])
        for row in report["curve"]:
            curves.setdefault(row["poisoned_rows"], []).append(row["certified_accuracy"])
    certified = [
        {
            "poisoned_rows": budget,
            "certified_accuracy": round(mean(values), 4),
            "stderr": round(stderr(values), 4),
        }
        for budget, values in sorted(curves.items())
    ]
    payload = {
        "rows": rows,
        "certified": certified,
        "ensemble_accuracy": round(mean(accuracies), 4),
        "shards_empirical": 16,
        "shards_certified": 32,
        "rows_per_shard": int(round(size * 0.7 / 16)),
        "seeds": seeds,
        "corpus_size": size,
    }
    save("partition", payload)
    return payload


def experiment_evasion(seeds, size):
    announce("evasive trigger shapes over %d seeds" % len(seeds))
    shapes = [
        ("plain ascii", "qz7x"),
        ("cyrillic homoglyph", "\u0430dmin"),
        ("invisible character", "qz\u200b7x"),
        ("bidi wrapped", "\u202eqz7x\u202c"),
    ]
    detectors = ["gram_purity", "contradiction", "rarity", "confusable"]
    rows = []
    for label, trigger in shapes:
        collected = {name: [] for name in detectors}
        for seed in seeds:
            corpus, train, _ = _prepare_split(seed, size)
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": trigger,
                    "target_label": "allow",
                    "poison_rate": 0.03,
                    "selection": "confident",
                },
                seed=seed % (2**31 - 1),
            )
            result = attack.poison(train)
            context = DefenseContext(
                labels=corpus.labels, target_label="allow", seed=1, budget=0.05
            )
            for name in detectors:
                report = build_defense(name).run(result.dataset, context)
                collected[name].append(report.metrics.get("auc") or 0.5)
        row = {"trigger": label, "trials": len(seeds)}
        row.update({name: round(mean(values), 4) for name, values in collected.items()})
        rows.append(row)
    payload = {"rows": rows, "detectors": detectors, "seeds": seeds, "corpus_size": size}
    save("evasion", payload)
    return payload


def experiment_precision(seeds, size):
    announce("measurement precision over %d seeds" % len(seeds))
    confident, chance, cdas = [], [], []
    for seed in seeds:
        corpus, train, test = _prepare_split(seed, size)
        for selection, sink in (("confident", confident), ("random", chance)):
            attack = build_attack(
                {
                    "kind": "backdoor",
                    "trigger": "qz7x",
                    "target_label": "allow",
                    "poison_rate": 0.02,
                    "selection": selection,
                },
                seed=seed % (2**31 - 1),
            )
            result = attack.poison(train)
            model, _ = train_surrogate(
                result.dataset, _member_config(), label_space=corpus.labels
            )
            evaluation = evaluate_model(model, test, attack, labels=corpus.labels)
            sink.append(evaluation.attack_success_rate)
            if selection == "confident":
                cdas.append(evaluation.clean_accuracy)
    differences = [a - b for a, b in zip(confident, chance)]
    asr_sd = stdev(confident)
    cda_sd = stdev(cdas)
    paired_sd = stdev(differences)
    unpaired_sd = math.sqrt(stdev(confident) ** 2 + stdev(chance) ** 2)
    needed = [
        {
            "half_width": target,
            "seeds_for_asr": max(2, int(math.ceil((1.96 * asr_sd / target) ** 2))),
            "seeds_for_cda": max(2, int(math.ceil((1.96 * cda_sd / target) ** 2))),
        }
        for target in (0.10, 0.05, 0.03, 0.02, 0.01)
    ]
    payload = {
        "seeds": len(seeds),
        "corpus_size": size,
        "asr_mean": round(mean(confident), 4),
        "asr_sd": round(asr_sd, 4),
        "cda_mean": round(mean(cdas), 4),
        "cda_sd": round(cda_sd, 4),
        "paired_difference": round(mean(differences), 4),
        "paired_sd": round(paired_sd, 4),
        "unpaired_sd": round(unpaired_sd, 4),
        "pairing_gain": round(unpaired_sd / paired_sd, 2) if paired_sd else None,
        "precision": needed,
        "permutation": paired_permutation_test(confident, chance, iterations=20000, seed=1),
    }
    save("precision", payload)
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
        "partition",
        "evasion",
        "precision",
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
        "every family, which is the argument for running the suite and fusing the ranks. "
        "The confusable scanner sits at 0.500 on every row because this corpus contains no "
        "lookalike or invisible characters at all: it is silent by design rather than weak, "
        "and section 10 is where it earns its place. A silent detector contributes nothing to "
        "the fused rank, which is why the ensemble row is unchanged by its presence."
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
    partition_payload = results.get("partition")
    if partition_payload:
        lines.append("## 9. Voting over disjoint shards")
        lines.append("")
        lines.append(
            "A partition ensemble splits the training set into disjoint shards by a hash of the "
            "record id, trains one model per shard and predicts by plurality vote. Poison in one "
            "shard cannot reach the others, so its influence is bounded by construction rather "
            "than by a detector getting lucky. Corpus size %d, %d seeds, %d shards, roughly %d "
            "training rows per shard."
            % (
                partition_payload["corpus_size"],
                len(partition_payload["seeds"]),
                partition_payload["shards_empirical"],
                partition_payload.get("rows_per_shard", 0),
            )
        )
        lines.append("")
        rows = []
        for row in partition_payload["rows"]:
            rows.append(
                [
                    "%.1f%%" % (100 * row["poison_rate"]),
                    row["trials"],
                    "%.3f ± %.3f" % (row["single_asr"], row["single_stderr"]),
                    "%.3f ± %.3f" % (row["ensemble_asr"], row["ensemble_stderr"]),
                    "%.0f%%" % (100 * row["reduction"]),
                    "%.4f" % row["accuracy_cost"],
                ]
            )
        lines.append(
            table(
                ["poison", "trials", "single model ASR", "ensemble ASR", "reduction", "accuracy cost"],
                rows,
            )
        )
        lines.append("")
        lines.append(
            "The reduction grows with the attack, which is the opposite of how a detector behaves. "
            "A larger attack has to spread across more shards to keep working, and every shard it "
            "touches is one vote, not a share of one model. The trade is shard size: each member "
            "only sees a fraction of the data, so on a corpus too small to leave roughly fifty "
            "rows per shard the members get weak and the vote loses more than the attacker does."
        )
        lines.append("")
        lines.append(
            "With %d shards the vote also carries a certificate. If the winning label leads the "
            "runner up by more than twice the number of corrupted shards, no attacker holding that "
            "many rows can change the answer, whatever those rows contain. Plain ensemble accuracy "
            "is %.4f."
            % (partition_payload["shards_certified"], partition_payload["ensemble_accuracy"])
        )
        lines.append("")
        lines.append(
            table(
                ["poisoned rows", "certified accuracy"],
                [
                    [row["poisoned_rows"], "%.4f ± %.4f" % (row["certified_accuracy"], row["stderr"])]
                    for row in partition_payload["certified"]
                ],
            )
        )
        lines.append("")
        lines.append(
            "Read this as a floor, not a score. Certificates cover handfuls of rows, so they are "
            "the right tool against a small deliberate insertion and the wrong tool against a "
            "vendor shipping two percent poison. The empirical reduction above is what covers "
            "that case."
        )
        lines.append("")

    evasion_payload = results.get("evasion")
    if evasion_payload:
        lines.append("## 10. Triggers built to survive human review")
        lines.append("")
        lines.append(
            "A trigger does not have to look strange. These four carry the same attack, but three "
            "of them render on screen as ordinary text: a Cyrillic letter that draws like a Latin "
            "one, a zero width character inside a word, and a bidirectional override. Detector AUC "
            "over %d seeds at a 3%% poison rate."
            % len(evasion_payload["seeds"])
        )
        lines.append("")
        headers = ["trigger"] + evasion_payload["detectors"]
        rows = [
            [row["trigger"]] + ["%.3f" % row[name] for name in evasion_payload["detectors"]]
            for row in evasion_payload["rows"]
        ]
        lines.append(table(headers, rows))
        lines.append("")
        lines.append(
            "The statistical scanners were never fooled, because they do not read the trigger, "
            "they count it. The reviewer is the one who gets fooled, which is why the confusable "
            "scanner exists and why every token it reports is printed with its code points "
            "expanded."
        )
        lines.append("")

    precision_payload = results.get("precision")
    if precision_payload:
        lines.append("## 11. How precise is any of this")
        lines.append("")
        lines.append(
            "Attack success moves from seed to seed because the corpus, the split and the victim "
            "rows all move with it. Over %d seeds at a 2%% budget on a %d document corpus, ASR has "
            "a standard deviation of **%.4f** and clean accuracy **%.4f**."
            % (
                precision_payload["seeds"],
                precision_payload["corpus_size"],
                precision_payload["asr_sd"],
                precision_payload["cda_sd"],
            )
        )
        lines.append("")
        lines.append(
            table(
                ["target half width", "seeds for ASR", "seeds for CDA"],
                [
                    [
                        "±%.2f" % row["half_width"],
                        row["seeds_for_asr"],
                        row["seeds_for_cda"],
                    ]
                    for row in precision_payload["precision"]
                ],
            )
        )
        lines.append("")
        lines.append(
            "That table is why every comparison in this study is paired on the seed rather than "
            "run as two independent groups. The difference between two strategies on the same "
            "seed has a standard deviation of %.4f against %.4f for the unpaired contrast, which "
            "is **%.1f times tighter** and needs roughly %d times fewer seeds for the same "
            "confidence. The strategy gap measured here is %.4f at p = %.5f."
            % (
                precision_payload["paired_sd"],
                precision_payload["unpaired_sd"],
                precision_payload["pairing_gain"] or 1.0,
                round((precision_payload["pairing_gain"] or 1.0) ** 2),
                precision_payload["paired_difference"],
                precision_payload["permutation"]["p_value"],
            )
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
    precision_seeds = list(spaced_seeds(int(config["seed"]), 8 if args.quick else 24, "precision"))

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
            "partition": lambda: experiment_partition(audit_seeds, audit_size),
            "evasion": lambda: experiment_evasion(seeds[: max(3, seed_count // 2)], audit_size),
            "precision": lambda: experiment_precision(precision_seeds, audit_size),
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
    results["partition"] = experiment_partition(audit_seeds, audit_size)
    results["evasion"] = experiment_evasion(seeds[: max(3, seed_count // 2)], audit_size)
    results["precision"] = experiment_precision(precision_seeds, audit_size)

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
