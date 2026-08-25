#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import { renderDocument, detectKind } from "../lib/render.mjs";

const USAGE = `plsv - render a poisonlab json report to a self contained html page

usage:
  node viewer/bin/plsv.mjs <report.json> [--out <file.html>] [--stdout]

options:
  --out <file>   destination file, defaults to the input name with an html suffix
  --stdout       write the page to stdout instead of a file
  --help         show this message
`;

function parseArguments(argv) {
  const options = { input: null, out: null, stdout: false };
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (item === "--help" || item === "-h") return { help: true };
    if (item === "--stdout") {
      options.stdout = true;
    } else if (item === "--out" || item === "-o") {
      index += 1;
      options.out = argv[index];
    } else if (!options.input) {
      options.input = item;
    }
  }
  return options;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help || !options.input) {
    process.stdout.write(USAGE);
    process.exit(options.help ? 0 : 1);
  }
  const inputPath = resolve(options.input);
  let payload;
  try {
    payload = JSON.parse(await readFile(inputPath, "utf8"));
  } catch (error) {
    process.stderr.write(`cannot read ${inputPath}: ${error.message}\n`);
    process.exit(2);
  }
  const kind = detectKind(payload);
  if (kind === "unknown") {
    process.stderr.write("this file does not look like a poisonlab report\n");
    process.exit(3);
  }
  const html = renderDocument(payload);
  if (options.stdout) {
    process.stdout.write(html);
    return;
  }
  const target = options.out
    ? resolve(options.out)
    : join(
        resolve(inputPath, ".."),
        `${basename(inputPath, extname(inputPath))}.html`
      );
  await writeFile(target, html, "utf8");
  process.stdout.write(`${target}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});
