from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import __version__, accel
from .analysis.potency import estimate_potency
from .data.loaders import load_source
from .data.splits import SplitPlan, stratified_split
from .data.versioning import DatasetStore
from .defenses.base import DefenseContext
from .defenses.suite import removal_report, run_suite, sanitize, stealth_summary
from .evaluate.evaluator import evaluate_model
from .forge.attacks import build_attack
from .seeding import derive_seed
from .train.engine import TrainConfig, train_model


def environment_fingerprint() -> Dict[str, Any]:
    return {
        "poisonlab": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "accelerator": accel.status().get("backend"),
    }


@dataclass
class RunPaths:
    root: str
    datasets: str = ""
    checkpoints: str = ""
    store: str = ""

    def prepare(self) -> "RunPaths":
        self.datasets = os.path.join(self.root, "datasets")
        self.checkpoints = os.path.join(self.root, "checkpoints")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.datasets, exist_ok=True)
        return self


@dataclass
class CampaignResult:
    config: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    paths: Optional[RunPaths] = None

    def save(self) -> str:
        if self.paths is None:
            raise RuntimeError("campaign has no output directory")
        path = os.path.join(self.paths.root, "report.json")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path


def _run_directory(config: Dict[str, Any]) -> RunPaths:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = str(config.get("name", "campaign"))
    root = os.path.join(str(config.get("output", "runs")), "%s-%s" % (name, stamp))
    return RunPaths(root=root).prepare()


def run_campaign(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    progress=None,
) -> CampaignResult:
    started = time.time()
    seed = int(config.get("seed", 1234))
    paths = RunPaths(root=output_dir).prepare() if output_dir else _run_directory(config)
    log = progress or (lambda message: None)
    timings: Dict[str, float] = {}

    log("loading data")
    mark = time.time()
    data_config = dict(config.get("data", {}))
    store_path = data_config.pop("store", os.path.join(paths.root, "store"))
    split_config = data_config.pop("split", {"train": 0.7, "validation": 0.1, "test": 0.2})
    corpus = load_source(data_config, seed=derive_seed(seed, "corpus") % (2**31 - 1))
    store = DatasetStore(store_path)
    clean_version = store.commit(corpus, tag="%s.clean" % config.get("name", "campaign"))
    plan = SplitPlan(
        train=float(split_config.get("train", 0.7)),
        validation=float(split_config.get("validation", 0.1)),
        test=float(split_config.get("test", 0.2)),
    )
    parts = stratified_split(corpus, plan, seed=derive_seed(seed, "split") % (2**31 - 1))
    train_set, test_set = parts["train"], parts["test"]
    validation_set = parts["validation"]
    timings["data"] = time.time() - mark

    log("forging poison")
    mark = time.time()
    attack_config = dict(config.get("attack", {}))
    attack = build_attack(attack_config, seed=derive_seed(seed, "attack") % (2**31 - 1))
    attack_result = attack.poison(train_set)
    poisoned_train = attack_result.dataset
    poisoned_version = store.commit(
        poisoned_train,
        tag="%s.poisoned" % config.get("name", "campaign"),
        parent=clean_version.digest,
        transform=attack.name,
    )
    timings["forge"] = time.time() - mark

    log("estimating potency")
    mark = time.time()
    carrier = None
    if hasattr(attack, "trigger_tokens"):
        carrier = attack.trigger_tokens()
    elif getattr(attack, "triggers", None):
        carrier = list(attack.triggers)
    elif getattr(attack, "concept_words", None):
        carrier = list(attack.concept_words)
    potency = estimate_potency(
        poisoned_train,
        attack_result.poisoned_uids,
        target_label=attack.target_label,
        carrier_tokens=carrier,
    )
    timings["potency"] = time.time() - mark

    train_config = TrainConfig.from_dict(config.get("train", {}))
    train_config.seed = derive_seed(seed, "train") % (2**31 - 1)
    labels = corpus.labels

    baseline_model = None
    baseline_evaluation = None
    if config.get("report", {}).get("baseline", True):
        log("training clean baseline")
        mark = time.time()
        baseline_model, _, _ = train_model(train_set, train_config, label_space=labels)
        baseline_evaluation = evaluate_model(baseline_model, test_set, labels=labels)
        timings["baseline"] = time.time() - mark

    log("training on poisoned data")
    mark = time.time()
    model, training_log, isolation = train_model(
        poisoned_train,
        train_config,
        label_space=labels,
        checkpoint_dir=paths.checkpoints,
    )
    timings["train"] = time.time() - mark

    log("evaluating")
    mark = time.time()
    evaluation = evaluate_model(
        model,
        test_set,
        attack,
        baseline_model=baseline_model,
        labels=labels,
    )
    timings["evaluate"] = time.time() - mark

    certification: Optional[Dict[str, Any]] = None
    if getattr(model, "shards", None):
        log("certifying the partition ensemble")
        mark = time.time()
        from .defenses.partition import certified_report

        certification = certified_report(model, test_set)
        timings["certify"] = time.time() - mark

    defense_payload: Dict[str, Any] = {"enabled": False}
    defense_config = dict(config.get("defense", {}))
    if defense_config.get("enabled", True):
        log("running defenses")
        mark = time.time()
        context = DefenseContext(
            labels=labels,
            target_label=attack.target_label,
            seed=derive_seed(seed, "defense") % (2**31 - 1),
            budget=float(defense_config.get("budget", 0.05)),
            max_n=int(defense_config.get("max_n", 2)),
            min_count=int(defense_config.get("min_count", 4)),
        )
        detectors = defense_config.get("detectors") or None
        reports, fused = run_suite(poisoned_train, context, detectors)
        stealth = stealth_summary(list(reports) + [fused], evaluation.attack_success_rate)
        defense_payload = {
            "enabled": True,
            "budget": context.budget,
            "detectors": [report.to_dict() for report in reports],
            "ensemble": fused.to_dict(),
            "stealth": stealth,
        }
        if defense_config.get("sanitize", True):
            log("sanitising and retraining")
            cleaned, removed = sanitize(poisoned_train, fused.scores, context.budget)
            removal = removal_report(poisoned_train, removed)
            repaired_model, _, _ = train_model(cleaned, train_config, label_space=labels)
            repaired = evaluate_model(
                repaired_model, test_set, attack, baseline_model=baseline_model, labels=labels
            )
            defense_payload["sanitised"] = {
                "removal": removal,
                "evaluation": repaired.to_dict(),
                "residual_asr": round(repaired.attack_success_rate, 6),
                "asr_reduction": round(
                    evaluation.attack_success_rate - repaired.attack_success_rate, 6
                ),
                "accuracy_cost": round(evaluation.clean_accuracy - repaired.clean_accuracy, 6),
            }
        timings["defense"] = time.time() - mark

    if config.get("report", {}).get("keep_datasets", True):
        poisoned_train.to_jsonl(os.path.join(paths.datasets, "train.poisoned.jsonl"))
        test_set.to_jsonl(os.path.join(paths.datasets, "test.clean.jsonl"))
        validation_set.to_jsonl(os.path.join(paths.datasets, "validation.clean.jsonl"))

    timings["total"] = time.time() - started
    report: Dict[str, Any] = {
        "name": config.get("name", "campaign"),
        "created_at": time.time(),
        "seed": seed,
        "environment": environment_fingerprint(),
        "config": config,
        "data": {
            "clean": clean_version.to_dict(),
            "poisoned": poisoned_version.to_dict(),
            "splits": {
                "train": len(train_set),
                "validation": len(validation_set),
                "test": len(test_set),
            },
            "labels": labels,
        },
        "attack": {"spec": attack.to_dict(), "result": attack_result.to_dict()},
        "potency": potency.to_dict(),
        "training": training_log.to_dict(),
        "isolation": isolation,
        "evaluation": evaluation.to_dict(),
        "certification": certification,
        "defense": defense_payload,
        "timings": {key: round(value, 4) for key, value in timings.items()},
    }
    if baseline_evaluation is not None:
        report["baseline"] = baseline_evaluation.to_dict()
    result = CampaignResult(config=config, report=report, paths=paths)
    result.save()
    return result


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number == number else fallback


def summarize(report: Dict[str, Any]) -> List[str]:
    evaluation = report.get("evaluation", {})
    attack = report.get("attack", {}).get("result", {})
    potency = report.get("potency", {})
    lines = [
        "campaign      : %s" % report.get("name"),
        "attack        : %s" % report.get("attack", {}).get("spec", {}).get("kind"),
        "poisoned      : %d records (%.3f%% of train)"
        % (_number(attack.get("applied", 0)), 100 * _number(attack.get("effective_rate", 0.0))),
        "predicted ASR : %.3f (potency index)" % _number(potency.get("predicted_asr", 0.0)),
        "measured ASR  : %.3f" % _number(evaluation.get("attack_success_rate", 0.0)),
        "clean accuracy: %.4f" % _number(evaluation.get("clean_accuracy", 0.0)),
    ]
    if report.get("baseline"):
        lines.append("baseline acc  : %.4f" % _number(report["baseline"].get("clean_accuracy", 0.0)))
    certification = report.get("certification")
    if certification:
        lines.append(
            "certified     : %d shards, median radius %d rows, %.4f accuracy certified against 1 row"
            % (
                certification.get("shards", 0),
                certification.get("median_radius", 0),
                next(
                    (
                        row["certified_accuracy"]
                        for row in certification.get("curve", [])
                        if row["poisoned_rows"] == 1
                    ),
                    0.0,
                ),
            )
        )
    stealth = report.get("defense", {}).get("stealth")
    if stealth:
        lines.append(
            "best detector : %s (recall %.2f at %.0f%% budget)"
            % (
                stealth.get("best_detector"),
                _number(stealth.get("best_recall_at_budget", 0.0)),
                100 * _number(report.get("defense", {}).get("budget", 0.05)),
            )
        )
        lines.append("stealth ASR   : %.3f" % _number(stealth.get("stealth_adjusted_asr", 0.0)))
    sanitised = report.get("defense", {}).get("sanitised")
    if sanitised:
        lines.append(
            "after cleanup : ASR %.3f (down %.3f), accuracy cost %.4f"
            % (
                _number(sanitised.get("residual_asr", 0.0)),
                _number(sanitised.get("asr_reduction", 0.0)),
                _number(sanitised.get("accuracy_cost", 0.0)),
            )
        )
    return lines
