// Full Accurate pipeline parity: JS (normalizer.js + student.js + accurate.js on onnxruntime-web) vs Python
const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const ort = require("onnxruntime-web"); ort.env.wasm.numThreads = 1;
const N = require(path.join(ROOT, "runtime", "normalizer.js")); const ST = require(path.join(ROOT, "runtime", "student.js"));
const EC = require(path.join(ROOT, "runtime", "encheck.js")); const A = require(path.join(ROOT, "runtime", "accurate.js"));
(async () => {
  const load = async (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "data", n), "utf8"));
  await N.init(load, EC);
  const S = await ST.create(ort, async (n) => fs.readFileSync(path.join(ROOT, "models", n)).buffer.slice(0), async (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "models", n), "utf8")), N._.lower, process.env.NEURAL_DIR || "student_B");
  A.init(N, S, ST);
  const P = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_accurate_python.json"), "utf8"));
  let same = 0; const ex = []; const t0 = Date.now();
  for (const [op, t, py] of P) {
    const r = await A.process(op, t);
    if (r.output === py.output && r.method === py.method) same++; else if (ex.length < 8) ex.push({ t, js: [r.output, r.method], py: [py.output, py.method] });
  }
  console.log(`accurate parity: ${same}/${P.length} identical (output + method) | ${((Date.now() - t0) / P.length).toFixed(1)} ms/item`);
  for (const e of ex) console.log(JSON.stringify(e));
})();
