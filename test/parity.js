// Run the JS runtime on test/parity_inputs.json and compare with test/parity_python.json
const fs = require("fs"), path = require("path");
const ROOT = path.join(__dirname, "..");
const N = require(path.join(ROOT, "runtime", "normalizer.js"));
(async () => {
  await N.init((name) => JSON.parse(fs.readFileSync(path.join(ROOT, "data", name), "utf8")));
  const inputs = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_inputs.json"), "utf8"));
  const py = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_python.json"), "utf8"));
  const t0 = Date.now(); const js = N.normalize(inputs); const ms = Date.now() - t0;
  let diff = 0; const ex = [];
  js.forEach((r, i) => {
    const p = py[i];
    const same = r.output === p.output && r.method === p.method && r.script === p.script &&
      Math.abs(r.confidence - p.confidence) < 0.0011 &&
      JSON.stringify(r.language.map((x) => x[0])) === JSON.stringify(p.language.map((x) => x[0])) &&
      r.language.every((x, j) => Math.abs(x[1] - p.language[j][1]) < 0.0011);
    if (!same) { diff++; if (ex.length < 12) ex.push({ in: inputs[i], js: [r.output, r.method, r.language], py: [p.output, p.method, p.language] }); }
  });
  console.log(`parity: ${inputs.length - diff}/${inputs.length} identical, ${diff} differ | JS time ${ms} ms (${(ms / inputs.length).toFixed(2)} ms/item)`);
  for (const e of ex) console.log(JSON.stringify(e));
  process.exit(diff ? 1 : 0);
})();
