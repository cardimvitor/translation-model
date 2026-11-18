// Accurate (JS runtime, onnxruntime-web) on the gold labels and the ABO real names; writes outputs for scoring by eval_js.py.
const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const ort = require("onnxruntime-web"); ort.env.wasm.numThreads = 1;
const N = require(path.join(ROOT, "runtime", "normalizer.js")), ST = require(path.join(ROOT, "runtime", "student.js"));
const EC = require(path.join(ROOT, "runtime", "encheck.js")); const A = require(path.join(ROOT, "runtime", "accurate.js"));
(async () => {
  await N.init(async (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "data", n), "utf8")), EC);
  const dir = process.env.NEURAL_DIR || "student_V6";
  const S = await ST.create(ort, async (n) => fs.readFileSync(path.join(ROOT, "models", n)).buffer.slice(0), async (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "models", n), "utf8")), N._.lower, dir);
  A.init(N, S, ST);
  const G = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "gold_products.json"), "utf8")).items;
  const sets = { gold: G.flatMap((g) => Object.values(g.src).map((s) => [g.en, s])) };
  const abo = path.join(ROOT, "build", "work", "abo_test.jsonl");
  if (fs.existsSync(abo)) sets.abo = fs.readFileSync(abo, "utf8").trim().split("\n").map((l) => JSON.parse(l)).map((p) => [[p.en], p.src]);
  const out = {};
  for (const [k, items] of Object.entries(sets)) {
    const t0 = Date.now(); out[k] = [];
    for (const [refs, s] of items) out[k].push([refs, s, (await A.process("translate", s)).output, (await S.translate(s))[0]]);
    out[k + "_ms"] = (Date.now() - t0) / items.length / 2;
  }
  fs.writeFileSync(path.join(ROOT, "test", "results_js_" + dir + ".json"), JSON.stringify(out));
})();
