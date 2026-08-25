import assert from "node:assert/strict";
import test from "node:test";
import { renderDocument } from "../lib/render.mjs";
import { barChart, doseCurve, scatter } from "../lib/charts.mjs";
import { escapeHtml, safeColour, safeToken } from "../lib/format.mjs";

const PAYLOADS = [
  '<script>alert(1)</script>',
  '"><script>alert(1)</script>',
  "<img src=x onerror=alert(1)>",
  "</style><script>alert(1)</script>",
  "</title><script>alert(1)</script>",
  "</svg><script>alert(1)</script>",
  "</text><script>alert(1)</script>",
  '" onmouseover="alert(1)',
  "' onfocus='alert(1)",
  "javascript:alert(1)",
  "<iframe src=javascript:alert(1)>",
  "<svg/onload=alert(1)>",
  "<body onload=alert(1)>",
  "<a href=\"javascript:alert(1)\">x</a>",
  "&lt;script&gt;alert(1)&lt;/script&gt;",
  "<!--<script>alert(1)</script>-->",
  "<style>@import 'javascript:alert(1)';</style>",
  "expression(alert(1))",
  "<script>alert(1)</script>",
];

const ALLOWED_TAGS = new Set([
  "!doctype",
  "html",
  "head",
  "meta",
  "title",
  "style",
  "body",
  "main",
  "h1",
  "h2",
  "h3",
  "p",
  "section",
  "div",
  "span",
  "code",
  "footer",
  "figure",
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "svg",
  "g",
  "rect",
  "text",
  "line",
  "circle",
  "path",
]);

function tagsOf(html) {
  return [...html.matchAll(/<\/?([a-zA-Z!][^\s/>]*)([^>]*)>/g)].map((match) => ({
    name: match[1].toLowerCase(),
    attributes: match[2],
    raw: match[0],
  }));
}

function assertInert(html, context) {
  const styleEnd = html.indexOf("</style>") + 8;
  const head = html.slice(0, styleEnd - 8);
  const body = html.slice(styleEnd);
  assert.equal(html.match(/<style>/g).length, 1, `${context}: unexpected extra style block`);
  assert.ok(!/<\/style>/i.test(head), `${context}: style closed early`);
  for (const tag of tagsOf(body)) {
    assert.ok(ALLOWED_TAGS.has(tag.name), `${context}: unexpected element <${tag.name}>`);
    assert.ok(!/\son\w+\s*=/i.test(tag.attributes), `${context}: event handler in ${tag.raw}`);
    assert.ok(!/javascript:/i.test(tag.attributes), `${context}: script url in ${tag.raw}`);
    assert.ok(!/expression\(/i.test(tag.attributes), `${context}: css expression in ${tag.raw}`);
  }
}

function campaignWith(payload) {
  return {
    name: payload,
    seed: payload,
    created_at: 1770000000,
    environment: { poisonlab: payload, accelerator: payload },
    attack: {
      spec: { kind: payload, params: { target_label: payload, selection: payload } },
      result: {
        applied: 84,
        requested: 84,
        requested_rate: payload,
        effective_rate: 0.02,
        details: { trigger: payload },
      },
    },
    data: { poisoned: { digest: payload } },
    potency: {
      carrier_tokens: [payload, payload],
      carrier_occurrences: payload,
      purity: 0.9,
      collision: 0,
      saliency: 0.2,
      contradiction: 0.9,
      effective_dose: 16,
      predicted_asr: 0.7,
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
          name: payload,
          seconds: 0.2,
          notes: payload,
          metrics: { auc: 1, average_precision: 1, recall_at_budget: 1, precision_at_budget: 0.5 },
          evidence: [{ gram: payload, count: 84, label: payload }],
        },
      ],
      ensemble: { name: payload, metrics: { auc: 0.99, recall_at_budget: 0.98 } },
      stealth: { best_detector: payload, best_recall_at_budget: 1, stealth_adjusted_asr: 0 },
      sanitised: {
        removal: { removed: 210, poison_recall: 1, clean_removed: 126 },
        residual_asr: 0.13,
        asr_reduction: 0.71,
        accuracy_cost: 0.001,
      },
    },
    timings: { [payload]: 4.1 },
  };
}

function sweepWith(payload) {
  return {
    axes: { [payload]: [payload, 0.01] },
    seeds: [payload, 2],
    rows: [
      { seed: payload, [payload]: 0.01, asr: 0.45, potency: 0.4, cda: 0.86 },
      { seed: 2, [payload]: 0.02, asr: 0.8, potency: 0.75, cda: 0.85 },
      { seed: 3, [payload]: 0.03, asr: 0.9, potency: 0.85, cda: 0.85 },
    ],
    groups: [
      {
        [payload]: payload,
        trials: 3,
        asr_mean: 0.2,
        asr_stderr: 0.02,
        lift_mean: 0.1,
        cda_mean: 0.86,
        potency_mean: 0.18,
      },
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
      key: payload,
      metric: "asr",
      reference: payload,
      comparisons: [
        { [payload]: payload, trials: 6, asr_mean: 0.6, difference: 0.2, relative: 0.5, p_value: 0.01 },
      ],
    },
    potency_correlation: { spearman: 0.93, samples: 3, mean_absolute_error: 0.05 },
  };
}

test("campaign pages neutralise every payload", () => {
  for (const payload of PAYLOADS) {
    const html = renderDocument(campaignWith(payload));
    assertInert(html, `campaign ${payload}`);
  }
});

test("sweep pages neutralise every payload", () => {
  for (const payload of PAYLOADS) {
    const html = renderDocument(sweepWith(payload));
    assertInert(html, `sweep ${payload}`);
  }
});

test("payload text still reaches the page in escaped form", () => {
  const html = renderDocument(campaignWith("<script>alert(1)</script>"));
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
});

test("the page ships no scriptable surface at all", () => {
  const html = renderDocument(campaignWith("plain"));
  assert.ok(!/<script/i.test(html));
  assert.ok(!/\son\w+\s*=/i.test(html.slice(html.indexOf("</style>"))));
  assert.ok(!/<a\s/i.test(html));
  assert.ok(!/href=/i.test(html));
});

test("tone and colour slots reject injected attributes", () => {
  assert.equal(safeToken('" onload="alert(1)'), "");
  assert.equal(safeToken("hot"), "hot");
  assert.equal(safeColour('red" onload="x', "var(--accent)"), "var(--accent)");
  assert.equal(safeColour("#ff0000", "var(--accent)"), "#ff0000");
  const chart = barChart([{ label: "x", value: 0.5, color: '" onload="alert(1)' }]);
  assert.ok(!/onload/i.test(chart));
});

test("chart primitives escape their labels", () => {
  const chart = barChart([{ label: "<script>alert(1)</script>", value: 0.5 }]);
  assert.ok(!/<script/i.test(chart));
  assert.match(chart, /&lt;script&gt;/);
  const points = scatter([{ x: 0.5, y: 0.5, label: "</svg><script>alert(1)</script>" }]);
  assert.ok(!/<script/i.test(points));
});

test("hostile numeric fields cannot break the svg", () => {
  const chart = doseCurve(
    [
      { rate: 0.01, asr: 0.5, stderr: "\" onload=\"alert(1)" },
      { rate: 0.02, asr: 0.8, stderr: 0.01 },
    ],
    { fitted: true, upper: 1, slope: 2, midpoint_log10: -2, critical_rate: 0.01 }
  );
  assert.ok(!/onload/i.test(chart));
});

test("prototype pollution attempts stay inert", () => {
  const payload = JSON.parse(
    '{"evaluation":{"clean_accuracy":0.5},"__proto__":{"polluted":"yes"},"name":"x"}'
  );
  const html = renderDocument(payload);
  assert.equal({}.polluted, undefined);
  assert.ok(html.length > 500);
});

test("missing and malformed sections do not throw", () => {
  const shapes = [
    { evaluation: {} },
    { evaluation: { clean_accuracy: null }, attack: null, defense: null },
    { evaluation: {}, potency: { carrier_tokens: "not-an-array" } },
    { evaluation: {}, defense: { enabled: true, detectors: [], ensemble: null } },
    { rows: [], groups: [] },
    { rows: [], groups: [], dose_response: { fitted: true, points: [] } },
  ];
  for (const shape of shapes) {
    assert.doesNotThrow(() => renderDocument(shape), JSON.stringify(shape));
  }
});

test("very large payloads stay bounded", () => {
  const html = renderDocument(campaignWith("A".repeat(20000)));
  assert.ok(html.length < 2_000_000);
  assertInert(html, "large payload");
});

test("escapeHtml is idempotent enough for nested rendering", () => {
  const once = escapeHtml("<b>&</b>");
  const twice = escapeHtml(once);
  assert.ok(!/<b>/.test(twice));
  assert.match(twice, /&amp;lt;b&amp;gt;/);
});
