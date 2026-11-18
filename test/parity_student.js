// Compare student outputs: JS (onnxruntime-web, wasm, same as phone) vs Python (onnxruntime) on test/parity_student_python.json
const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const ort = require("onnxruntime-web"); ort.env.wasm.numThreads = 1;
const N = require(path.join(ROOT, "runtime", "normalizer.js")); const ST = require(path.join(ROOT, "runtime", "student.js"));
(async () => {
  const P = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_student_python.json"), "utf8"));
  const S = await ST.create(ort, async (n) => fs.readFileSync(path.join(ROOT, "models", n)).buffer.slice(0), async (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "models", n), "utf8")), N._ ? (s) => s.toLowerCase().split("ß").join("ss").split("ς").join("σ") : null);
  let same = 0, conf = 0; const t0 = Date.now(); const ex = [];
  for (const [t, out, c] of P) {
    const [o, cc] = await S.translate(t);
    if (o === out) same++; else if (ex.length < 6) ex.push({ t, js: o, py: out });
    if (Math.abs(cc - c) < 0.011) conf++;
  }
  console.log(`student parity: ${same}/${P.length} identical outputs, ${conf}/${P.length} confidence within 0.01 | ${((Date.now() - t0) / P.length).toFixed(0)} ms/item (wasm, 1 thread)`);
  for (const e of ex) console.log(JSON.stringify(e));
})();
