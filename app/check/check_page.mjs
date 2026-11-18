// Checks the browser application against the package and captures its screenshot.
//
// The script opens the page in a Chromium-based browser, replays the requests of
// reference.json through the Python runtime of the page and compares every number with the
// CPython reference. The outcome is written to results/app_check.json. With --screenshot it
// then runs the LTE map with the delay spread of the automotive plant and saves an image of the result.
//
// Usage, from the repository root, with the page served locally or from its public address:
//   npm install puppeteer-core
//   node app/check/check_page.mjs <browser executable> <page address> [--screenshot <png file>]
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const TOLERANCE = 1e-6;
const [executable, address, ...rest] = process.argv.slice(2);
if (!executable || !address) {
  console.log("usage: node app/check/check_page.mjs <browser executable> <page address> [--screenshot <png file>]");
  process.exit(2);
}
const shotAt = rest.indexOf("--screenshot");
const shotFile = shotAt >= 0 ? rest[shotAt + 1] : null;

function compare(a, b, path, out) {
  if (typeof a === "number" && typeof b === "number") {
    const d = Math.abs(a - b);
    out.count += 1;
    out.largest = Math.max(out.largest, d);
    if (d > TOLERANCE * Math.max(1, Math.abs(a))) out.problems.push(path + ": " + a + " against " + b);
  } else if (a !== null && typeof a === "object" && b !== null && typeof b === "object") {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) compare(a[k], b[k], path + "." + k, out);
  } else if (a !== b) {
    out.problems.push(path + ": " + JSON.stringify(a) + " against " + JSON.stringify(b));
  }
}

async function ready(page) {
  await page.goto(address, { waitUntil: "load" });
  await page.waitForFunction(() => document.getElementById("status").textContent === "Ready.", { timeout: 300000 });
}

const browser = await puppeteer.launch({ executablePath: executable, headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 1180, height: 900, deviceScaleFactor: 4 });
await ready(page);

const reference = JSON.parse(readFileSync(join(here, "reference.json"), "utf8"));
const runtime = JSON.parse(await page.evaluate(() => window.ofdmestRuntime()));
runtime.pyodide = await page.evaluate(() => document.getElementById("pyVersion").textContent);
runtime.ofdmest = await page.evaluate(() => document.getElementById("pkgVersion").textContent);
runtime.browser = await browser.version();
const out = { count: 0, largest: 0, problems: [] };
for (let i = 0; i < reference.cases.length; i++) {
  const c = reference.cases[i];
  const result = JSON.parse(await page.evaluate((q) => window.ofdmestRunPoint(q), JSON.stringify(c.request)));
  compare(c.result, result, "case" + (i + 1), out);
  console.log("case " + (i + 1) + " of " + reference.cases.length + " (" + c.request.grid + ", " + c.request.snr_db + " dB) replayed");
}
const record = {
  page_runtime: runtime,
  reference_runtime: reference.environment,
  cases: reference.cases.length,
  numbers_compared: out.count,
  largest_absolute_difference: out.largest,
  tolerance: TOLERANCE,
  differences_beyond_tolerance: out.problems.length,
};
writeFileSync(join(root, "results", "app_check.json"), JSON.stringify(record, null, 1) + String.fromCharCode(10));
console.log(out.count + " numbers compared, largest absolute difference " + out.largest.toExponential(2));
for (const p of out.problems) console.log("DIFFERENT " + p);

if (shotFile && out.problems.length === 0) {
  await ready(page);
  // The pictured case is the LTE map with the delay spread of the automotive plant, 131.8 ns.
  const started = Date.now();
  await page.evaluate(() => { document.getElementById("spread").value = "131.8"; document.getElementById("form").requestSubmit(); });
  await page.waitForFunction(() => document.getElementById("status").textContent === "Done.", { timeout: 600000 });
  console.log("pictured case computed in " + ((Date.now() - started) / 1000).toFixed(1) + " s");
  const bottom = await page.evaluate(() => {
    const cards = document.querySelectorAll("#results > section");
    const form = document.getElementById("form").getBoundingClientRect().bottom;
    return Math.max(form, cards[1].getBoundingClientRect().bottom) + window.scrollY + 16;
  });
  await page.screenshot({ path: shotFile, clip: { x: 0, y: 0, width: 1180, height: Math.ceil(bottom) } });
  console.log("screenshot written to " + shotFile);
}
await browser.close();
process.exit(out.problems.length === 0 ? 0 : 1);
