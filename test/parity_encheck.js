const fs = require("fs"), path = require("path"); const ROOT = path.join(__dirname, "..");
const C = require(path.join(ROOT, "runtime", "encheck.js"));
(async () => {
  await C.init((n) => JSON.parse(fs.readFileSync(path.join(ROOT, "data", n), "utf8")));
  const P = JSON.parse(fs.readFileSync(path.join(ROOT, "test", "parity_encheck_python.json"), "utf8"));
  let diff = 0; const t0 = process.hrtime.bigint();
  P.inputs.forEach((t, i) => {
    for (const [j, mode] of [[0, "strict"], [1, "relaxed"]]) {
      const r = C.isEnglish(t, mode), p = P.res[i][j];
      if (r.english !== p.english || Math.abs(r.score - p.score) > 0.0011 || r.reason !== p.reason) { diff++; if (diff < 6) console.log(JSON.stringify({ t, mode, js: r, py: p })); }
    }
  });
  const us = Number(process.hrtime.bigint() - t0) / 1000 / (P.inputs.length * 2);
  console.log(`encheck parity: ${P.inputs.length * 2 - diff}/${P.inputs.length * 2} identical | JS ${us.toFixed(1)} µs/call`);
})();
