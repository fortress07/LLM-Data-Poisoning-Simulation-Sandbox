from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..analysis.potency import estimate_potency
from ..config import merge
from ..data.loaders import load_source
from ..data.record import Dataset
from ..data.splits import SplitPlan, stratified_split
from ..defenses.base import DefenseContext
from ..defenses.suite import run_suite, stealth_summary
from ..evaluate.evaluator import evaluate_model
from ..evaluate.statistics import (
    fit_dose_response,
    mean,
    paired_permutation_test,
    spearman,
    stderr,
)
from ..forge.attacks import build_attack
from ..seeding import derive_seed
from ..train.engine import TrainConfig, train_model

_CACHE: Dict[Any, Any] = {}


def _prepare(config: Dict[str, Any], seed: int) -> Tuple[Dataset, Dataset, List[str], Any]:
    key = (seed, repr(config.get("data")), repr(config.get("train")))
    if key in _CACHE:
        return _CACHE[key]
    data_config = dict(config.get("data", {}))
    data_config.pop("store", None)
    split_config = data_config.pop("split", {"train": 0.7, "validation": 0.1, "test": 0.2})
    corpus = load_source(data_config, seed=derive_seed(seed, "corpus") % (2**31 - 1))
    plan = SplitPlan(
        train=float(split_config.get("train", 0.7)),
        validation=float(split_config.get("validation", 0.1)),
        test=float(split_config.get("test", 0.2)),
    )
    parts = stratified_split(corpus, plan, seed=derive_seed(seed, "split") % (2**31 - 1))
    train_config = TrainConfig.from_dict(config.get("train", {}))
    train_config.seed = derive_seed(seed, "train") % (2**31 - 1)
    train_config.checkpoint_every = 0
    baseline, _, _ = train_model(parts["train"], train_config, label_space=corpus.labels)
    payload = (parts["train"], parts["test"], corpus.labels, baseline)
    _CACHE[key] = payload
    return payload


def clear_cache() -> None:
    _CACHE.clear()


def _set_dotted(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    parts = key.split(".")
    patch: Dict[str, Any] = {}
    node = patch
    for part in parts[:-1]:
        node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return merge(config, patch)


def trial(
    config: Dict[str, Any],
    seed: int,
    include_defense: bool = False,
    defense_budget: float = 0.05,
) -> Dict[str, Any]:
    started = time.time()
    train_set, test_set, labels, baseline = _prepare(config, seed)
    attack = build_attack(
        dict(config.get("attack", {})), seed=derive_seed(seed, "attack") % (2**31 - 1)
    )
    result = attack.poison(train_set)
    train_config = TrainConfig.from_dict(config.get("train", {}))
    train_config.seed = derive_seed(seed, "train") % (2**31 - 1)
    train_config.checkpoint_every = 0
    model, _, _ = train_model(result.dataset, train_config, label_space=labels)
    evaluation = evaluate_model(model, test_set, attack, baseline_model=baseline, labels=labels)
    carrier = attack.trigger_tokens() if hasattr(attack, "trigger_tokens") else None
    if carrier is None and getattr(attack, "triggers", None):
        carrier = list(attack.triggers)
    potency = estimate_potency(
        result.dataset,
        result.poisoned_uids,
        target_label=attack.target_label,
        carrier_tokens=carrier,
    )
    row: Dict[str, Any] = {
        "seed": seed,
        "attack": attack.name,
        "poison_rate": result.rate,
        "effective_rate": round(result.effective_rate, 6),
        "poisoned": result.applied,
        "asr": round(evaluation.attack_success_rate, 6),
        "asr_ci": [round(value, 6) for value in evaluation.attack_success_ci],
        "baseline_asr": round(evaluation.baseline_success_rate or 0.0, 6),
        "lift": round(evaluation.attack_lift, 6),
        "cda": round(evaluation.clean_accuracy, 6),
        "cda_drop": round(evaluation.accuracy_drop or 0.0, 6),
        "potency": round(potency.potency_index, 6),
        "effective_dose": round(potency.effective_dose, 6),
        "seconds": round(time.time() - started, 4),
    }
    if include_defense:
        context = DefenseContext(
            labels=labels,
            target_label=attack.target_label,
            seed=derive_seed(seed, "defense") % (2**31 - 1),
            budget=defense_budget,
        )
        reports, fused = run_suite(result.dataset, context)
        stealth = stealth_summary(list(reports) + [fused], evaluation.attack_success_rate)
        row["detection"] = {
            report.name: {
                "auc": report.metrics.get("auc"),
                "recall_at_budget": report.metrics.get("recall_at_budget"),
            }
            for report in list(reports) + [fused]
        }
        row["stealth"] = stealth
    return row


def sweep(
    config: Dict[str, Any],
    axes: Dict[str, Sequence[Any]],
    seeds: Sequence[int],
    include_defense: bool = False,
    defense_budget: float = 0.05,
    progress=None,
) -> Dict[str, Any]:
    keys = list(axes)
    combinations = list(itertools.product(*[list(axes[key]) for key in keys])) if keys else [()]
    rows: List[Dict[str, Any]] = []
    total = len(combinations) * len(seeds)
    done = 0
    for values in combinations:
        variant = config
        for key, value in zip(keys, values):
            variant = _set_dotted(variant, key, value)
        for seed in seeds:
            row = trial(variant, int(seed), include_defense, defense_budget)
            for key, value in zip(keys, values):
                row[key] = value
            rows.append(row)
            done += 1
            if progress is not None:
                progress(done, total, row)
    return {
        "axes": {key: list(axes[key]) for key in keys},
        "seeds": list(seeds),
        "rows": rows,
        "groups": group_rows(rows, keys),
    }


def group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        signature = tuple(row.get(key) for key in keys)
        buckets.setdefault(signature, []).append(row)
    groups: List[Dict[str, Any]] = []
    for signature, members in buckets.items():
        entry: Dict[str, Any] = {key: value for key, value in zip(keys, signature)}
        entry["trials"] = len(members)
        for metric in ("asr", "lift", "cda", "cda_drop", "potency", "poisoned", "effective_rate"):
            values = [float(member[metric]) for member in members if metric in member]
            if values:
                entry["%s_mean" % metric] = round(mean(values), 6)
                entry["%s_stderr" % metric] = round(stderr(values), 6)
        detections = [member.get("stealth", {}) for member in members if member.get("stealth")]
        if detections:
            entry["detection_recall_mean"] = round(
                mean([float(item.get("best_recall_at_budget", 0.0)) for item in detections]), 6
            )
            entry["stealth_adjusted_asr_mean"] = round(
                mean([float(item.get("stealth_adjusted_asr", 0.0)) for item in detections]), 6
            )
        groups.append(entry)
    groups.sort(key=lambda item: [str(item.get(key)) for key in keys])
    return groups


def dose_response(result: Dict[str, Any], rate_key: str = "attack.poison_rate") -> Dict[str, Any]:
    groups = [group for group in result.get("groups", []) if rate_key in group]
    rates = [float(group[rate_key]) for group in groups]
    responses = [float(group.get("asr_mean", 0.0)) for group in groups]
    fit = fit_dose_response(rates, responses)
    fit["points"] = [
        {"rate": rate, "asr": response, "stderr": group.get("asr_stderr", 0.0)}
        for rate, response, group in zip(rates, responses, groups)
    ]
    return fit


def compare(
    result: Dict[str, Any],
    key: str,
    metric: str = "asr",
    reference: Optional[Any] = None,
) -> Dict[str, Any]:
    rows = result.get("rows", [])
    values: Dict[Any, Dict[Any, float]] = {}
    for row in rows:
        if key not in row:
            continue
        signature = tuple(
            (axis, row.get(axis)) for axis in result.get("axes", {}) if axis != key
        ) + (("seed", row.get("seed")),)
        values.setdefault(row[key], {})[signature] = float(row.get(metric, 0.0))
    variants = sorted(values, key=lambda item: str(item))
    if reference is None:
        reference = "random" if "random" in variants else variants[0]
    baseline = values.get(reference, {})
    comparisons: List[Dict[str, Any]] = []
    for variant in variants:
        current = values[variant]
        shared = sorted(set(current) & set(baseline), key=str)
        left = [current[signature] for signature in shared]
        right = [baseline[signature] for signature in shared]
        test = paired_permutation_test(left, right, iterations=5000, seed=17)
        comparisons.append(
            {
                key: variant,
                "trials": len(shared),
                "%s_mean" % metric: round(mean(left), 6),
                "reference_mean": round(mean(right), 6),
                "difference": round(test["difference"], 6),
                "relative": round(
                    test["difference"] / mean(right) if mean(right) else 0.0, 6
                ),
                "p_value": round(test["p_value"], 6),
            }
        )
    comparisons.sort(key=lambda item: -item["%s_mean" % metric])
    return {"key": key, "metric": metric, "reference": reference, "comparisons": comparisons}


def potency_correlation(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result.get("rows", [])
    predicted = [float(row.get("potency", 0.0)) for row in rows]
    measured = [float(row.get("asr", 0.0)) for row in rows]
    if len(rows) < 3:
        return {"samples": len(rows)}
    errors = [abs(a - b) for a, b in zip(predicted, measured)]
    return {
        "samples": len(rows),
        "spearman": round(spearman(predicted, measured), 6),
        "mean_absolute_error": round(mean(errors), 6),
    }


def flatten_rows(
    result: Dict[str, Any], columns: Optional[Iterable[str]] = None
) -> List[List[Any]]:
    rows = result.get("rows", [])
    if not rows:
        return []
    simple = {key for row in rows for key in row if not isinstance(row[key], (dict, list))}
    keys = list(columns or sorted(simple))
    table = [keys]
    for row in rows:
        table.append([row.get(key) for key in keys])
    return table
