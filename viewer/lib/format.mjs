const CONTROL_PATTERN = new RegExp("[\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f]", "g");

export const escapeHtml = (value) =>
  String(value ?? "")
    .replace(CONTROL_PATTERN, "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const TOKEN_PATTERN = /^[A-Za-z0-9_-]{0,32}$/;

export const safeToken = (value, fallback = "") => {
  const text = String(value ?? "");
  return TOKEN_PATTERN.test(text) ? text : fallback;
};

const COLOUR_PATTERN = /^(#[0-9a-fA-F]{3,8}|var\(--[a-zA-Z0-9-]{1,32}\))$/;

export const safeColour = (value, fallback) => {
  const text = String(value ?? "");
  return COLOUR_PATTERN.test(text) ? text : fallback;
};

export const asArray = (value) => (Array.isArray(value) ? value : []);

export const isFinite = (value) => typeof value === "number" && Number.isFinite(value);

export const percent = (value, digits = 1) =>
  isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";

export const fixed = (value, digits = 3) => (isFinite(value) ? value.toFixed(digits) : "n/a");

export const compact = (value) => {
  if (!isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  if (Math.abs(value) >= 0.001) return value.toFixed(4);
  return value.toExponential(1);
};

export const titleCase = (value) =>
  String(value ?? "")
    .split(/[_\s.]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

export const timestamp = (value) => {
  if (!isFinite(value)) return "";
  const date = new Date(value * 1000);
  return date.toISOString().replace("T", " ").slice(0, 19);
};
