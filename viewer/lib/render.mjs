import { barChart, doseCurve, scatter } from "./charts.mjs";
import {
  asArray,
  compact,
  escapeHtml,
  fixed,
  isFinite,
  percent,
  safeToken,
  timestamp,
  titleCase,
} from "./format.mjs";

const STYLE = `
:root {
  color-scheme: light dark;
  --bg: #f6f7fb;
  --panel: #ffffff;
  --ink: #16181d;
  --muted: #5b6172;
  --line: #e2e5ee;
  --accent: #3d5afe;
  --accent-2: #d81b60;
  --accent-3: #00897b;
  --accent-4: #f4a300;
  --danger: #d32f2f;
  --ok: #2e7d32;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101216;
    --panel: #181b21;
    --ink: #eef1f6;
    --muted: #99a0b0;
    --line: #262b34;
    --accent: #7c8cff;
    --accent-2: #ff6ea9;
    --accent-3: #35c2ac;
    --accent-4: #ffc247;
    --danger: #ff6b6b;
    --ok: #6ddf8a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif;
}
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 27px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 34px 0 12px; }
h3 { font-size: 15px; margin: 20px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
p { margin: 8px 0; }
.sub { color: var(--muted); margin-bottom: 22px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.card .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; }
.card .v { font-size: 25px; font-weight: 600; margin-top: 6px; }
.card .n { color: var(--muted); font-size: 12px; margin-top: 4px; }
.card.hot .v { color: var(--danger); }
.card.cool .v { color: var(--ok); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code { background: color-mix(in srgb, var(--line) 60%, transparent); padding: 1px 5px; border-radius: 4px; font-size: 13px; }
.scroll { overflow-x: auto; }
.chart svg { width: 100%; height: auto; }
figure { margin: 8px 0 0; }
.bar-label { fill: var(--ink); font-size: 12px; }
.bar-number { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.bar-track { fill: color-mix(in srgb, var(--line) 70%, transparent); }
.grid { stroke: var(--line); stroke-width: 1; }
.axis, .axis-title, .marker-label { fill: var(--muted); font-size: 11px; }
.axis-line { stroke: var(--muted); stroke-width: 1; opacity: 0.5; }
.curve { fill: none; stroke: var(--accent); stroke-width: 2.5; }
.dot { fill: var(--accent-2); }
.error { stroke: var(--accent-2); stroke-width: 2; opacity: 0.45; }
.marker { stroke: var(--accent-4); stroke-width: 1.5; stroke-dasharray: 4 4; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); color: var(--muted); margin-right: 6px; }
footer { color: var(--muted); font-size: 12px; margin-top: 34px; text-align: center; }
`;

const card = (label, value, note = "", tone = "") =>
  `<div class="card ${safeToken(tone)}"><div class="k">${escapeHtml(label)}</div><div class="v">${escapeHtml(value)}</div>${
    note ? `<div class="n">${escapeHtml(note)}</div>` : ""
  }</div>`;

const tableOf = (headers, rows) => {
  if (!rows.length) return "";
  const head = headers
    .map((item) => `<th class="${item.numeric ? "num" : ""}">${escapeHtml(item.label ?? item)}</th>`)
    .join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${row
          .map(
            (cell, index) =>
              `<td class="${headers[index] && headers[index].numeric ? "num" : ""}">${
                cell === null || cell === undefined ? "" : escapeHtml(cell)
              }</td>`
          )
          .join("")}</tr>`
    )
    .join("");
  return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
};

function campaignSections(report) {
  const evaluation = report.evaluation ?? {};
  const attack = report.attack ?? {};
  const spec = attack.spec ?? {};
  const result = attack.result ?? {};
  const potency = report.potency ?? {};
  const defense = report.defense ?? {};
  const baseline = report.baseline ?? null;
  const parts = [];

  parts.push(`<div class="cards">
    ${card("attack success", percent(evaluation.attack_success_rate), `${evaluation.probe_size ?? 0} probes`, "hot")}
    ${card("clean accuracy", percent(evaluation.clean_accuracy), baseline ? `baseline ${percent(baseline.clean_accuracy)}` : "")}
    ${card("poison budget", percent(result.effective_rate, 3), `${result.applied ?? 0} records`)}
    ${card("predicted ASR", percent(potency.predicted_asr), "potency index, no training")}
    ${card(
      "stealth ASR",
      percent(defense.stealth ? defense.stealth.stealth_adjusted_asr : null),
      defense.stealth ? `best detector ${defense.stealth.best_detector ?? "none"}` : "",
      defense.stealth && defense.stealth.stealth_adjusted_asr > 0.3 ? "hot" : "cool"
    )}
  </div>`);

  parts.push(`<section class="panel">
    <h2>Attack configuration</h2>
    ${tableOf(
      ["field", "value"],
      [
        ["kind", spec.kind],
        ["target label", (spec.params ?? {}).target_label],
        ["selection", (spec.params ?? {}).selection ?? "random"],
        ["requested rate", (result.requested_rate ?? 0).toString()],
        ["records poisoned", `${result.applied ?? 0} of ${result.requested ?? 0} requested`],
        ["dataset digest", (report.data?.poisoned?.digest ?? "").slice(0, 16)],
      ]
    )}
  </section>`);

  const comparison = [
    { label: "attack success rate", value: evaluation.attack_success_rate },
    { label: "baseline model ASR", value: evaluation.baseline_success_rate },
    { label: "clean accuracy", value: evaluation.clean_accuracy },
    { label: "predicted ASR", value: potency.predicted_asr },
  ];
  parts.push(`<section class="panel"><h2>Outcome</h2>${barChart(comparison, { label: "outcome" })}</section>`);

  if (potency && Object.keys(potency).length) {
    parts.push(`<section class="panel">
      <h2>Potency signals</h2>
      <p class="sub">Estimated before any training from corpus statistics alone.</p>
      ${tableOf(
        ["signal", { label: "value", numeric: true }],
        [
          ["carrier tokens", asArray(potency.carrier_tokens).join(", ")],
          ["carrier occurrences", potency.carrier_occurrences],
          ["label purity", fixed(potency.purity, 4)],
          ["collision", fixed(potency.collision, 4)],
          ["saliency", fixed(potency.saliency, 4)],
          ["contradiction", fixed(potency.contradiction, 4)],
          ["effective dose", compact(potency.effective_dose)],
        ]
      )}
    </section>`);
  }

  if (defense.enabled) {
    const detectors = [...asArray(defense.detectors)];
    if (defense.ensemble) detectors.push(defense.ensemble);
    const rows = detectors.map((detector) => [
      detector.name,
      fixed(detector.metrics?.auc, 3),
      fixed(detector.metrics?.average_precision, 3),
      percent(detector.metrics?.recall_at_budget),
      percent(detector.metrics?.precision_at_budget),
      `${fixed(detector.seconds, 2)}s`,
    ]);
    parts.push(`<section class="panel">
      <h2>Detection</h2>
      <p class="sub">Review budget ${percent(defense.budget)} of the training set.</p>
      ${barChart(
        detectors.map((detector) => ({
          label: detector.name,
          value: detector.metrics?.recall_at_budget ?? 0,
        })),
        { label: "recall at budget" }
      )}
      ${tableOf(
        [
          "detector",
          { label: "auc", numeric: true },
          { label: "ap", numeric: true },
          { label: "recall", numeric: true },
          { label: "precision", numeric: true },
          { label: "time", numeric: true },
        ],
        rows
      )}
    </section>`);

    const evidence = asArray(defense.detectors)
      .flatMap((detector) => asArray(detector.evidence).slice(0, 3).map((item) => [detector.name, JSON.stringify(item)]))
      .slice(0, 12);
    if (evidence.length) {
      parts.push(`<section class="panel"><h2>Top signals</h2>${tableOf(["detector", "evidence"], evidence)}</section>`);
    }

    if (defense.sanitised) {
      const removal = defense.sanitised.removal ?? {};
      parts.push(`<section class="panel">
        <h2>After sanitising</h2>
        ${tableOf(
          ["metric", { label: "value", numeric: true }],
          [
            ["records removed", removal.removed],
            ["poison recall", percent(removal.poison_recall)],
            ["clean records lost", removal.clean_removed],
            ["residual ASR", percent(defense.sanitised.residual_asr)],
            ["ASR reduction", percent(defense.sanitised.asr_reduction)],
            ["accuracy cost", percent(defense.sanitised.accuracy_cost)],
          ]
        )}
      </section>`);
    }
  }

  const timings = Object.entries(report.timings ?? {});
  if (timings.length) {
    parts.push(`<section class="panel"><h2>Timings</h2>${tableOf(
      ["stage", { label: "seconds", numeric: true }],
      timings.map(([key, value]) => [key, fixed(value, 3)])
    )}</section>`);
  }
  return parts.join("\n");
}

function sweepSections(result) {
  const parts = [];
  const axes = Object.keys(result.axes ?? {});
  const groups = asArray(result.groups);
  const rows = asArray(result.rows);
  const best = groups.reduce((carry, group) => (group.asr_mean > (carry?.asr_mean ?? -1) ? group : carry), null);

  parts.push(`<div class="cards">
    ${card("configurations", String(groups.length), `${rows.length} trials`)}
    ${card("seeds", String(asArray(result.seeds).length))}
    ${card("peak ASR", percent(best?.asr_mean), best ? axes.map((axis) => `${axis}=${best[axis]}`).join(", ") : "")}
    ${card(
      "potency correlation",
      fixed(result.potency_correlation?.spearman, 3),
      `spearman over ${result.potency_correlation?.samples ?? 0} trials`
    )}
  </div>`);

  if (result.dose_response && result.dose_response.fitted) {
    const fit = result.dose_response;
    parts.push(`<section class="panel">
      <h2>Dose response</h2>
      <p class="sub">Fitted logistic in log rate space, r squared ${fixed(fit.r_squared, 3)}.</p>
      ${doseCurve(asArray(fit.points), fit)}
      ${tableOf(
        ["parameter", { label: "value", numeric: true }],
        [
          ["ceiling", fixed(fit.upper, 3)],
          ["slope", fixed(fit.slope, 3)],
          ["critical rate", percent(fit.critical_rate, 3)],
          ["rate for ASR 50", percent(fit.rate_for_asr_50, 3)],
          ["rate for ASR 90", percent(fit.rate_for_asr_90, 3)],
        ]
      )}
    </section>`);
  }

  if (groups.length) {
    const headers = [...axes, { label: "trials", numeric: true }, { label: "asr", numeric: true }, { label: "lift", numeric: true }, { label: "cda", numeric: true }, { label: "potency", numeric: true }];
    const body = groups.map((group) => [
      ...axes.map((axis) => group[axis]),
      group.trials,
      `${fixed(group.asr_mean, 3)} ± ${fixed(group.asr_stderr, 3)}`,
      fixed(group.lift_mean, 3),
      fixed(group.cda_mean, 4),
      fixed(group.potency_mean, 3),
    ]);
    parts.push(`<section class="panel"><h2>Grid</h2>${tableOf(headers, body)}</section>`);
  }

  if (result.comparison) {
    const key = result.comparison.key;
    const body = asArray(result.comparison.comparisons).map((entry) => [
      entry[key],
      entry.trials,
      fixed(entry[`${result.comparison.metric}_mean`], 3),
      fixed(entry.difference, 3),
      percent(entry.relative),
      fixed(entry.p_value, 4),
    ]);
    parts.push(`<section class="panel">
      <h2>Comparison against ${escapeHtml(String(result.comparison.reference))}</h2>
      ${tableOf(
        [key, { label: "trials", numeric: true }, { label: "mean", numeric: true }, { label: "difference", numeric: true }, { label: "relative", numeric: true }, { label: "p value", numeric: true }],
        body
      )}
    </section>`);
  }

  const points = rows
    .filter((row) => isFinite(row.potency) && isFinite(row.asr))
    .map((row) => ({ x: row.potency, y: row.asr, label: `seed ${row.seed}` }));
  if (points.length > 2) {
    parts.push(`<section class="panel">
      <h2>Predicted against measured</h2>
      <p class="sub">Each point is one trial, the diagonal is a perfect prediction.</p>
      ${scatter(points, { xLabel: "potency index", yLabel: "measured ASR" })}
    </section>`);
  }
  return parts.join("\n");
}

export function detectKind(payload) {
  if (payload && Array.isArray(payload.rows) && payload.groups) return "sweep";
  if (payload && payload.evaluation) return "campaign";
  return "unknown";
}

export function renderDocument(payload) {
  const kind = detectKind(payload);
  let title = "PoisonLab report";
  let subtitle = "";
  let body = "";
  if (kind === "campaign") {
    title = `Campaign: ${payload.name ?? "run"}`;
    const environment = payload.environment ?? {};
    subtitle = `seed ${payload.seed ?? "?"} · poisonlab ${environment.poisonlab ?? "?"} · ${environment.accelerator ?? "python"} kernels · ${timestamp(payload.created_at)}`;
    body = campaignSections(payload);
  } else if (kind === "sweep") {
    title = "Parameter sweep";
    subtitle = `${asArray(payload.rows).length} trials across ${asArray(payload.seeds).length} seeds`;
    body = sweepSections(payload);
  } else {
    body = `<section class="panel"><p>Unrecognised report shape.</p></section>`;
  }
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'">
<meta name="referrer" content="no-referrer">
<title>${escapeHtml(title)}</title>
<style>${STYLE}</style>
</head>
<body>
<main>
<h1>${escapeHtml(title)}</h1>
<p class="sub">${escapeHtml(subtitle)}</p>
${body}
<footer>Generated by the PoisonLab viewer. Research use only.</footer>
</main>
</body>
</html>
`;
}

export { titleCase };
