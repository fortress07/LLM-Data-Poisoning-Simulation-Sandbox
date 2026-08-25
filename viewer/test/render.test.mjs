import assert from "node:assert/strict";
import test from "node:test";
import { detectKind, renderDocument } from "../lib/render.mjs";
import { barChart, doseCurve, scatter } from "../lib/charts.mjs";
import { compact, escapeHtml, percent } from "../lib/format.mjs";

const campaign = {
  name: "unit",
  seed: 7,
  created_at: 1770000000,
  environment: { poisonlab: "0.4.0", accelerator: "native" },
  attack: {
    spec: { kind: "backdoor", params: { target_label: "allow", selection: "gain" } },
    result: { applied: 84, requested: 84, requested_rate: 0.02, effective_rate: 0.02 },
  },
  data: { poisoned: { digest: "abcdef0123456789" } },
  potency: {
    carrier_tokens: ["qz7x"],
    carrier_occurrences: 84,
    purity: 0.95,
    collision: 0.0,
    saliency: 0.22,
    contradiction: 0.9,
    effective_dose: 16.2,
    predicted_asr: 0.77,
  },
  evaluation: {
    clean_accuracy: 0.86,
    attack_success_rate: 0.84,
    baseline_success_rate: 0.15,
    probe_size: 420,
  },
  baseline: { clean_accuracy: 0.865 },
  defense: {
    enabled: true,
    budget: 0.05,
    detectors: [
      {
        name: "gram_purity",
        seconds: 0.2,
        metrics: { auc: 1, average_precision: 1, recall_at_budget: 1, precision_at_budget: 0.47 },
        evidence: [{ gram: "qz7x", count: 84 }],
      },
    ],
    ensemble: { name: "ensemble", metrics: { auc: 0.99, recall_at_budget: 0.98 } },
    stealth: { best_detector: "gram_purity", best_recall_at_budget: 1, stealth_adjusted_asr: 0 },
    sanitised: {
      removal: { removed: 210, poison_recall: 1, clean_removed: 126 },
      residual_asr: 0.13,
      asr_reduction: 0.71,
      accuracy_cost: 0.001,
    },
  },
  timings: { total: 4.1 },
};

const sweep = {
  axes: { "attack.poison_rate": [0.005, 0.01, 0.02] },
  seeds: [1, 2, 3],
  rows: [
    { seed: 1, "attack.poison_rate": 0.005, asr: 0.2, potency: 0.18, cda: 0.86 },
    { seed: 2, "attack.poison_rate": 0.01, asr: 0.45, potency: 0.4, cda: 0.86 },
    { seed: 3, "attack.poison_rate": 0.02, asr: 0.8, potency: 0.75, cda: 0.85 },
  ],
  groups: [
    { "attack.poison_rate": 0.005, trials: 3, asr_mean: 0.2, asr_stderr: 0.02, lift_mean: 0.1, cda_mean: 0.86, potency_mean: 0.18 },
    { "attack.poison_rate": 0.02, trials: 3, asr_mean: 0.8, asr_stderr: 0.03, lift_mean: 0.6, cda_mean: 0.85, potency_mean: 0.75 },
  ],
  dose_response: {
    fitted: true,
    upper: 0.95,
    slope: 3.2,
    midpoint_log10: -2,
    critical_rate: 0.01,
    r_squared: 0.98,
    points: [
      { rate: 0.005, asr: 0.2, stderr: 0.02 },
      { rate: 0.01, asr: 0.45, stderr: 0.03 },
      { rate: 0.02, asr: 0.8, stderr: 0.03 },
    ],
  },
  comparison: {
    key: "attack.selection",
    metric: "asr",
    reference: "random",
    comparisons: [
      { "attack.selection": "gain", trials: 6, asr_mean: 0.6, difference: 0.2, relative: 0.5, p_value: 0.01 },
    ],
  },
  potency_correlation: { spearman: 0.93, samples: 3, mean_absolute_error: 0.05 },
};

test("detects report kinds", () => {
  assert.equal(detectKind(campaign), "campaign");
  assert.equal(detectKind(sweep), "sweep");
  assert.equal(detectKind({}), "unknown");
});

test("renders a campaign page", () => {
  const html = renderDocument(campaign);
  assert.match(html, /<!doctype html>/);
  assert.match(html, /Campaign: unit/);
  assert.match(html, /gram_purity/);
  assert.match(html, /qz7x/);
  assert.ok(html.length > 3000);
});

test("renders a sweep page with a curve", () => {
  const html = renderDocument(sweep);
  assert.match(html, /Parameter sweep/);
  assert.match(html, /Dose response/);
  assert.match(html, /path class="curve"/);
});

test("escapes hostile text", () => {
  const html = renderDocument({
    ...campaign,
    name: '<img src=x onerror="alert(1)">',
  });
  assert.ok(!html.includes("<img src=x"));
  assert.match(html, /&lt;img src=x/);
});

test("chart helpers stay silent on empty input", () => {
  assert.equal(barChart([]), "");
  assert.equal(doseCurve([], null), "");
  assert.equal(scatter([]), "");
});

test("formatting helpers", () => {
  assert.equal(percent(0.5), "50.0%");
  assert.equal(percent(null), "n/a");
  assert.equal(compact(0.0001), "1.0e-4");
  assert.equal(escapeHtml('<a href="x">'), "&lt;a href=&quot;x&quot;&gt;");
});
