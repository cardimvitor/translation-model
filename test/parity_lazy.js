// Lazy loading must give the same answers: init with preload "none", then normalizeAsync per small batch.
const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const N = require(path.join(ROOT, "runtime", "normalizer.js")); const EC = require(path.join(ROOT, "runtime", "encheck.js"));
(async () => {
  const loaded = [];
  await N.init(async (n) => { loaded.push(n); return JSON.parse(fs.readFileSync(path.join(ROOT, "data", n), "utf8")); }, EC, { preload: "none" });
  const P = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_python.json"), "utf8"));
  const lat = P.filter((r) => /^[\x00-\x7f]*$/.test(r.input)).slice(0, 50);
  await N.normalizeAsync(lat.map((r) => [r.operation, r.input]));
  console.log("after a Latin-only batch, loaded:", loaded.filter((n) => n.startsWith("gazetteer/") || n.startsWith("translit/")).join(", "));
  let same = 0;
  for (let i = 0; i < P.length; i += 25) {
    const batch = P.slice(i, i + 25); const out = await N.normalizeAsync(batch.map((r) => [r.operation, r.input]));
    batch.forEach((r, j) => { if (JSON.stringify(out[j]) === JSON.stringify(r)) same++; else console.log("diff", JSON.stringify(r), JSON.stringify(out[j])); });
  }
  console.log(`lazy parity: ${same}/${P.length} identical`);
  try { N.normalize([["translate", 5]]); } catch (e) { console.log("bad input ->", e.name, e.code); }
})();
