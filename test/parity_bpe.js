// Word-piece tokenizer parity: runtime/bpe.js vs runtime/bpe.py on every gold source and test case.
// python test/make_parity_bpe.py <bpe.json> && node test/parity_bpe.js
const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const BPE = require(path.join(ROOT, "runtime", "bpe.js"));
const P = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_bpe_python.json"), "utf8"));
const B = JSON.parse(fs.readFileSync(P.bpe, "utf8"));
let same = 0; const ex = [];
for (const [side, text, py] of P.items) {
  const js = BPE.encode(text, B[side].merges);
  if (JSON.stringify(js) === JSON.stringify(py)) same++; else if (ex.length < 5) ex.push({ side, text, js, py });
}
console.log(`bpe parity: ${same}/${P.items.length} identical`); for (const e of ex) console.log(JSON.stringify(e));
process.exitCode = same === P.items.length ? 0 : 1;
