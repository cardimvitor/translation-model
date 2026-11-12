// JS SentencePiece (runtime/spm.js) must give the same pieces as Python (runtime/spm.py).
const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const SPM = require(path.join(ROOT, "runtime", "spm.js")); const N = require(path.join(ROOT, "runtime", "normalizer.js"));
const D = path.join(ROOT, "models", process.env.NEURAL_DIR || "student_V6");
const sp = SPM.create(JSON.parse(fs.readFileSync(path.join(D, "spm_src.json"), "utf8")));
const prep = (s) => N._.lower(s.normalize("NFKC")).replace(/\s+/g, " ").trim();
const P = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_spm_python.json"), "utf8"));
let same = 0; const ex = [];
for (const [t, py] of P) { const js = sp.encode(prep(t)); if (JSON.stringify(js) === JSON.stringify(py)) same++; else if (ex.length < 5) ex.push({ t, js, py }); }
console.log(`spm parity: ${same}/${P.length} identical`); ex.forEach((e) => console.log(JSON.stringify(e)));
process.exit(same === P.length ? 0 : 1);
