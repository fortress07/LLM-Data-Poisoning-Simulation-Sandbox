from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..safety import sanitize_terminal


def _cell(item: Any) -> str:
    if item is None:
        return ""
    return sanitize_terminal(item).replace("|", chr(92) + "|")


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(_cell(item) for item in headers) + " |"]
    lines.append("|" + "|".join([":--"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    return "\n".join(lines)


def _percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.2f%%" % (100.0 * float(value))


def render_campaign(report: Dict[str, Any]) -> str:
    attack = report.get("attack", {})
    spec = attack.get("spec", {})
    result = attack.get("result", {})
    evaluation = report.get("evaluation", {})
    potency = report.get("potency", {})
    defense = report.get("defense", {})
    lines: List[str] = []
    lines.append("# Campaign report: %s" % report.get("name", "campaign"))
    lines.append("")
    lines.append("Seed `%s`, accelerator `%s`, poisonlab `%s`." % (
        report.get("seed"),
        report.get("environment", {}).get("accelerator"),
        report.get("environment", {}).get("poisonlab"),
    ))
    lines.append("")
    lines.append("## Attack")
    lines.append("")
    lines.append(
        table(
            ["field", "value"],
            [
                ["kind", spec.get("kind")],
                ["target label", spec.get("params", {}).get("target_label")],
                ["requested rate", result.get("requested_rate")],
                ["applied", result.get("applied")],
                ["effective rate", _percent(result.get("effective_rate"))],
                ["selection", spec.get("params", {}).get("selection", "random")],
            ],
        )
    )
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    rows = [
        ["clean accuracy (CDA)", _percent(evaluation.get("clean_accuracy"))],
        ["attack success rate (ASR)", _percent(evaluation.get("attack_success_rate"))],
        ["ASR 95% interval", str(evaluation.get("attack_success_ci"))],
        ["baseline ASR", _percent(evaluation.get("baseline_success_rate"))],
        ["attack lift", _percent(evaluation.get("attack_lift"))],
        ["predicted ASR (potency index)", _percent(potency.get("predicted_asr"))],
    ]
    if report.get("baseline"):
        rows.append(["baseline accuracy", _percent(report["baseline"].get("clean_accuracy"))])
        rows.append(["accuracy drop", _percent(evaluation.get("accuracy_drop"))])
    lines.append(table(["metric", "value"], rows))
    lines.append("")
    if potency:
        lines.append("## Potency estimate")
        lines.append("")
        lines.append(
            table(
                ["signal", "value"],
                [
                    ["carrier tokens", ", ".join(potency.get("carrier_tokens", []))],
                    ["carrier occurrences", potency.get("carrier_occurrences")],
                    ["label purity", round(potency.get("purity", 0.0), 4)],
                    ["collision", round(potency.get("collision", 0.0), 4)],
                    ["saliency", round(potency.get("saliency", 0.0), 4)],
                    ["contradiction", round(potency.get("contradiction", 0.0), 4)],
                    ["effective dose", round(potency.get("effective_dose", 0.0), 3)],
                ],
            )
        )
        lines.append("")
    if defense.get("enabled"):
        lines.append("## Detection")
        lines.append("")
        rows = []
        for detector in defense.get("detectors", []):
            metrics = detector.get("metrics", {})
            rows.append(
                [
                    detector.get("name"),
                    metrics.get("auc"),
                    metrics.get("average_precision"),
                    _percent(metrics.get("recall_at_budget")),
                    _percent(metrics.get("precision_at_budget")),
                    "%.2fs" % detector.get("seconds", 0.0),
                ]
            )
        ensemble = defense.get("ensemble", {})
        if ensemble:
            metrics = ensemble.get("metrics", {})
            rows.append(
                [
                    "ensemble",
                    metrics.get("auc"),
                    metrics.get("average_precision"),
                    _percent(metrics.get("recall_at_budget")),
                    _percent(metrics.get("precision_at_budget")),
                    "",
                ]
            )
        lines.append(
            table(["detector", "AUC", "AP", "recall at budget", "precision", "time"], rows)
        )
        lines.append("")
        stealth = defense.get("stealth", {})
        if stealth:
            lines.append(
                "Best detector `%s` recovers %s of the poison at a %s review budget, "
                "leaving a stealth adjusted ASR of %s."
                % (
                    stealth.get("best_detector"),
                    _percent(stealth.get("best_recall_at_budget")),
                    _percent(defense.get("budget")),
                    _percent(stealth.get("stealth_adjusted_asr")),
                )
            )
            lines.append("")
        sanitised = defense.get("sanitised", {})
        if sanitised:
            removal = sanitised.get("removal", {})
            lines.append("## After sanitising")
            lines.append("")
            lines.append(
                table(
                    ["metric", "value"],
                    [
                        ["records removed", removal.get("removed")],
                        ["poison recall", _percent(removal.get("poison_recall"))],
                        ["clean records lost", removal.get("clean_removed")],
                        ["residual ASR", _percent(sanitised.get("residual_asr"))],
                        ["ASR reduction", _percent(sanitised.get("asr_reduction"))],
                        ["accuracy cost", _percent(sanitised.get("accuracy_cost"))],
                    ],
                )
            )
            lines.append("")
    timings = report.get("timings", {})
    if timings:
        lines.append("## Timings")
        lines.append("")
        lines.append(
            table(["stage", "seconds"], [[key, value] for key, value in sorted(timings.items())])
        )
        lines.append("")
    return sanitize_terminal("\n".join(lines))


def render_sweep(result: Dict[str, Any]) -> str:
    lines: List[str] = ["# Sweep report", ""]
    axes = result.get("axes", {})
    lines.append("Axes: %s" % ", ".join("`%s`" % key for key in axes) or "none")
    lines.append("")
    lines.append("Seeds: %s" % ", ".join(str(seed) for seed in result.get("seeds", [])))
    lines.append("")
    groups = result.get("groups", [])
    if groups:
        keys = [key for key in axes]
        headers = keys + ["trials", "ASR", "lift", "CDA", "potency"]
        rows = []
        for group in groups:
            rows.append(
                [group.get(key) for key in keys]
                + [
                    group.get("trials"),
                    "%.3f ± %.3f" % (group.get("asr_mean", 0.0), group.get("asr_stderr", 0.0)),
                    "%.3f" % group.get("lift_mean", 0.0),
                    "%.4f" % group.get("cda_mean", 0.0),
                    "%.3f" % group.get("potency_mean", 0.0),
                ]
            )
        lines.append(table(headers, rows))
        lines.append("")
    fit = result.get("dose_response")
    if fit and fit.get("fitted"):
        lines.append("## Dose response")
        lines.append("")
        lines.append(
            table(
                ["parameter", "value"],
                [
                    ["ceiling", fit.get("upper")],
                    ["slope", fit.get("slope")],
                    ["critical rate (ASR 50% of ceiling)", fit.get("critical_rate")],
                    ["rate for ASR 50%", fit.get("rate_for_asr_50")],
                    ["rate for ASR 90%", fit.get("rate_for_asr_90")],
                    ["r squared", fit.get("r_squared")],
                ],
            )
        )
        lines.append("")
    comparison = result.get("comparison")
    if comparison:
        lines.append("## Strategy comparison")
        lines.append("")
        key = comparison.get("key")
        rows = []
        for entry in comparison.get("comparisons", []):
            rows.append(
                [
                    entry.get(key),
                    entry.get("trials"),
                    "%.3f" % entry.get("asr_mean", entry.get("lift_mean", 0.0)),
                    "%+.3f" % entry.get("difference", 0.0),
                    "%+.1f%%" % (100 * entry.get("relative", 0.0)),
                    "%.4f" % entry.get("p_value", 1.0),
                ]
            )
        lines.append(
            table(
                [key, "trials", "mean", "difference", "relative", "p value"],
                rows,
            )
        )
        lines.append("")
    correlation = result.get("potency_correlation")
    if correlation and correlation.get("samples", 0) >= 3:
        lines.append(
            "Potency index versus measured ASR: Spearman %.3f over %d trials, "
            "mean absolute error %.3f."
            % (
                correlation.get("spearman", 0.0),
                correlation.get("samples", 0),
                correlation.get("mean_absolute_error", 0.0),
            )
        )
        lines.append("")
    return sanitize_terminal("\n".join(lines))
