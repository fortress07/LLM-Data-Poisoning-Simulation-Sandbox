import { asArray, escapeHtml, fixed, isFinite, percent, safeColour } from "./format.mjs";

const PALETTE = ["var(--accent)", "var(--accent-2)", "var(--accent-3)", "var(--accent-4)"];

export function barChart(series, options = {}) {
  const { width = 640, rowHeight = 30, max = 1, formatter = percent, label = "" } = options;
  const rows = asArray(series).filter((item) => isFinite(item.value));
  if (!rows.length) return "";
  const labelWidth = 150;
  const height = rows.length * rowHeight + 16;
  const trackWidth = width - labelWidth - 76;
  const bars = rows
    .map((item, index) => {
      const y = index * rowHeight + 8;
      const ratio = Math.max(0, Math.min(1, item.value / max));
      const barWidth = Math.max(2, ratio * trackWidth);
      const colour = safeColour(item.color, PALETTE[index % PALETTE.length]);
      return `
      <g>
        <text class="bar-label" x="0" y="${y + 15}">${escapeHtml(item.label)}</text>
        <rect class="bar-track" x="${labelWidth}" y="${y + 4}" width="${trackWidth}" height="16" rx="4"></rect>
        <rect class="bar-value" x="${labelWidth}" y="${y + 4}" width="${barWidth}" height="16" rx="4" fill="${colour}"></rect>
        <text class="bar-number" x="${labelWidth + trackWidth + 8}" y="${y + 16}">${escapeHtml(formatter(item.value))}</text>
      </g>`;
    })
    .join("");
  return `<figure class="chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)}">${bars}</svg></figure>`;
}

export function doseCurve(points, fit, options = {}) {
  const { width = 680, height = 320 } = options;
  const usable = asArray(points).filter((point) => isFinite(point.rate) && point.rate > 0);
  if (usable.length < 2) return "";
  const padding = { top: 20, right: 24, bottom: 46, left: 52 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const logs = usable.map((point) => Math.log10(point.rate));
  const minLog = Math.min(...logs) - 0.15;
  const maxLog = Math.max(...logs) + 0.15;
  const xOf = (value) => padding.left + ((value - minLog) / (maxLog - minLog)) * innerWidth;
  const yOf = (value) => padding.top + (1 - Math.max(0, Math.min(1, value))) * innerHeight;
  const gridY = [0, 0.25, 0.5, 0.75, 1]
    .map((value) => {
      const y = yOf(value);
      return `<line class="grid" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>
      <text class="axis" x="${padding.left - 10}" y="${y + 4}" text-anchor="end">${Math.round(value * 100)}%</text>`;
    })
    .join("");
  const gridX = usable
    .map((point) => {
      const x = xOf(Math.log10(point.rate));
      return `<line class="grid" x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}"></line>
      <text class="axis" x="${x}" y="${height - padding.bottom + 18}" text-anchor="middle">${(point.rate * 100).toFixed(2)}%</text>`;
    })
    .join("");
  let curve = "";
  if (fit && fit.fitted) {
    const steps = 80;
    const path = [];
    for (let index = 0; index <= steps; index += 1) {
      const logRate = minLog + ((maxLog - minLog) * index) / steps;
      const exponent = -fit.slope * (logRate - fit.midpoint_log10);
      const value = fit.upper / (1 + Math.exp(exponent));
      path.push(`${index === 0 ? "M" : "L"}${xOf(logRate).toFixed(2)},${yOf(value).toFixed(2)}`);
    }
    curve = `<path class="curve" d="${path.join(" ")}"></path>`;
    if (isFinite(fit.critical_rate)) {
      const criticalLog = Math.log10(fit.critical_rate);
      if (criticalLog >= minLog && criticalLog <= maxLog) {
        const x = xOf(criticalLog);
        curve += `<line class="marker" x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}"></line>
        <text class="marker-label" x="${x + 6}" y="${padding.top + 14}">critical ${(fit.critical_rate * 100).toFixed(2)}%</text>`;
      }
    }
  }
  const dots = usable
    .map((point) => {
      const x = xOf(Math.log10(point.rate));
      const y = yOf(point.asr);
      const error = isFinite(point.stderr) ? point.stderr : 0;
      const bar = error
        ? `<line class="error" x1="${x}" y1="${yOf(point.asr - error)}" x2="${x}" y2="${yOf(point.asr + error)}"></line>`
        : "";
      return `${bar}<circle class="dot" cx="${x}" cy="${y}" r="4"><title>rate ${(point.rate * 100).toFixed(3)}%, ASR ${fixed(point.asr)}</title></circle>`;
    })
    .join("");
  return `<figure class="chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="dose response curve">
    ${gridY}${gridX}
    <line class="axis-line" x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}"></line>
    <line class="axis-line" x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}"></line>
    ${curve}${dots}
    <text class="axis-title" x="${padding.left + innerWidth / 2}" y="${height - 8}" text-anchor="middle">poison rate</text>
  </svg></figure>`;
}

export function scatter(points, options = {}) {
  const { width = 620, height = 320, xLabel = "x", yLabel = "y" } = options;
  const usable = asArray(points).filter((point) => isFinite(point.x) && isFinite(point.y));
  if (!usable.length) return "";
  const padding = { top: 20, right: 20, bottom: 44, left: 50 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const xOf = (value) => padding.left + Math.max(0, Math.min(1, value)) * innerWidth;
  const yOf = (value) => padding.top + (1 - Math.max(0, Math.min(1, value))) * innerHeight;
  const diagonal = `<line class="grid" x1="${xOf(0)}" y1="${yOf(0)}" x2="${xOf(1)}" y2="${yOf(1)}"></line>`;
  const dots = usable
    .map(
      (point) =>
        `<circle class="dot" cx="${xOf(point.x)}" cy="${yOf(point.y)}" r="4"><title>${escapeHtml(point.label ?? "")} predicted ${fixed(point.x)}, measured ${fixed(point.y)}</title></circle>`
    )
    .join("");
  return `<figure class="chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="prediction against measurement">
    ${diagonal}
    <line class="axis-line" x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}"></line>
    <line class="axis-line" x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}"></line>
    ${dots}
    <text class="axis-title" x="${padding.left + innerWidth / 2}" y="${height - 8}" text-anchor="middle">${escapeHtml(xLabel)}</text>
    <text class="axis-title" transform="rotate(-90 14 ${padding.top + innerHeight / 2})" x="14" y="${padding.top + innerHeight / 2}" text-anchor="middle">${escapeHtml(yLabel)}</text>
  </svg></figure>`;
}
