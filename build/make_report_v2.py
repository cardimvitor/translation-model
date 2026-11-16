"""The report, rebuilt from scratch (Nov 15): one clean page with everything since the first version.
  python build/make_report_v2.py   -> web/report/index.html
Numbers of v11-v14 and of the places / mixed tests are read from build/work; older models use their final measured values."""
import html, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
E = html.escape


def load(p, d=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return d


# name, key, date, exact, chrF labels, chrF real names, model MB, label ms, name ms, commercial, one-line idea
M = [
    ("Fast (rules only)", "fast", "Nov 4–5", 20.0, 42.6, 39.1, "21 (data)", "0.3", "0.9", True, "Place list, address rules, transliteration, product word list. No neural network."),
    ("v3", "v3", "Nov 5–6", 21.9, 43.2, 33.6, "7", "~5", "~22", False, "First neural model: 7M parameters, characters, trained from nothing on 24k pairs written by NLLB."),
    ("v4 A", "A", "Nov 6–8", 28.1, 49.6, 37.7, "14", "~8", "~33", False, "13M parameters, 81k product titles translated by NLLB."),
    ("v4 B", "B", "Nov 6–8", 24.1, 46.5, 37.4, "14", "~9", "~32", True, "Same model, titles translated by OPUS-MT (commercial use allowed)."),
    ("v4 C", "C", "Nov 6–8", 30.6, 51.0, 38.4, "14", "~8", "~34", False, "Same model, both teachers together."),
    ("v5 B2", "B2", "Nov 8–10", 28.1, 49.8, 38.4, "22", "~13", "~50", True, "22M parameters; the teacher's own English answers added as extra targets."),
    ("v5 C2", "C2", "Nov 8–10", 29.1, 51.0, 38.6, "22", "~11", "~53", False, "As B2, with OPUS-MT and NLLB."),
    ("v5 H", "H", "Nov 8–10", 26.2, 48.8, 38.4, "22", "~12", "~52", True, "As B2, with answers from Hy-MT2."),
    ("v5 ALL3", "ALL3", "Nov 8–10", 26.9, 49.2, 38.5, "22", "~12", "~56", False, "All three teachers."),
    ("v5 HY", "HY", "Nov 8–10", 23.4, 45.5, 35.9, "22", "~13", "~56", True, "Hy-MT2 writes everything; only 8k pairs because it is slow."),
    ("v5 MAX", "MAX", "Nov 10", 28.4, 49.1, 45.7, "22", "11.9", "59.6", True, "Adds 35k real product names written by people (Amazon Berkeley Objects)."),
    ("JEPA-Gen", "JEPA", "Nov 10–11", 20.6, 43.5, 42.2, "22", "~11", "~59", True, "Encoder first trained to predict meaning vectors (JEPA), then a translator on top."),
    ("Baseline", "BASE", "Nov 10–11", 23.8, 45.9, 43.5, "22", "~12", "~65", True, "Same model and hours as JEPA-Gen, ordinary training, for a fair comparison."),
    ("v6", "V6", "Nov 11–12", 65.6, 79.6, 66.9, "106", "11.0", "63.7", True, "Starts from a pretrained translator (OPUS-MT mul-en, 77M parameters), fine-tuned on all product data."),
    ("v7 (MoE)", "V7", "Nov 12", 65.9, 80.8, 67.2, "106 + 3 per language", "14.5", "69.6", True, "v6 frozen, plus one small expert per language; only the detected language's expert is active."),
    ("v8 Complete", "V8", "Nov 13", 68.8, 83.1, 67.7, "57", "7.7", "28.2", True, "v6's encoder, a 2-layer decoder, English-only output layer, experts in the encoder only."),
    ("v8 Fast", "V8F", "Nov 13", 65.0, 79.7, 62.2, "42", "3.3", "17.2", True, "Cut from v8: 3 encoder layers, 1 decoder layer, no experts. For quick validation."),
    ("v9", "V9", "Nov 13–14", 72.5, 86.0, 67.2, "57", "5.8", "25.4", True, "v8 trained further on 200k more short labels, with the new post-training tool."),
    ("v10", "V10", "Nov 14", 72.8, 86.8, 67.3, "57", "6.3", "26.8", True, "Beam search (2 candidates) and common terms written by Hy-MT2 in all 16 languages."),
    ("v11", "V11", "Nov 14", 75.9, 87.6, 67.6, "57", "6.5", "26.8", True, "Training data checked by Hy-MT2-7B; English double check; reliable flag."),
    ("v12", "V12", "Nov 15", 76.2, 87.8, 67.5, "57", "6.5", "24.5", True, "v11's weights; a word check so one foreign word inside an English label is translated."),
    ("v13", "V13", "Nov 15", 76.2, 87.8, 67.6, "57", "6.1", "23.7", True, "v11's weights; place list of the whole world; address and transliteration fixes."),
    ("v14", "V14", "Nov 15", 76.2, 87.7, 67.5, "57", "6.1", "23.7", True, "v11's weights; villages, islands, districts; mined Arabic and Hindi readings; more street forms."),
]
FOLDER = {"v3": "student_v3", "A": "student_A", "B": "student_B", "C": "student_C", "B2": "student_B2", "C2": "student_C2", "H": "student_H", "ALL3": "student_ALL3", "HY": "student_HY", "MAX": "student_MAX",
          "JEPA": "student_JEPA", "BASE": "student_BASE", "V6": "student_V6", "V7": "student_V7", "V8": "student_V8", "V8F": "student_V8F", "V9": "student_V9", "V10": "student_V10", "V11": "student_V11",
          "V12": "student_V12", "V13": "student_V13", "V14": "student_V14"}
ZIP = {"v3": "en-normalizer-accurate-v3.zip", "A": "en-normalizer-accurate-A.zip", "B": "en-normalizer-accurate-B.zip", "C": "en-normalizer-accurate-C.zip", "B2": "en-normalizer-accurate-B2.zip", "C2": "en-normalizer-accurate-C2.zip",
       "H": "en-normalizer-accurate-H.zip", "ALL3": "en-normalizer-accurate-ALL3.zip", "HY": "en-normalizer-accurate-HY.zip", "MAX": "en-normalizer-accurate-MAX.zip", "JEPA": "en-normalizer-accurate-JEPA.zip", "BASE": "en-normalizer-accurate-BASE.zip",
       "V6": "en-normalizer-accurate-V6.zip", "V7": "en-normalizer-accurate-V7.zip", "V8": "label-normalizer-V8.zip", "V8F": "label-normalizer-V8F.zip", "V9": "label-normalizer-V9.zip", "V10": "label-normalizer-V10.zip", "V11": "label-normalizer-V11.zip",
       "V12": "label-normalizer-V12.zip", "V13": "label-normalizer-V13.zip", "V14": "label-normalizer-V14.zip", "fast": "en-normalizer-fast.zip"}
FP32 = {"V6", "V8", "V8F", "V9", "V10", "V11", "V12", "V13", "V14"}
REPORTS = {"B2": "v5", "C2": "v5", "H": "v5", "ALL3": "v5", "HY": "v5", "MAX": "v5", "V6": "v6", "V7": "v7", "V8": "v8", "V8F": "v8", "V9": "v9", "V10": "v10", "V11": "v11", "V12": "v12", "V13": "v13", "V14": "v14"}
GH = "https://github.com/cardimvitor/translation-model"; BR = "ccr-9202c40a-kuyiob"; T = f"{GH}/tree/{BR}"; B = f"{GH}/blob/{BR}"
A = lambda u, t: f'<a href="{u}">{E(t)}</a>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr" + (' class="hl"' if r and isinstance(r[0], str) and r[0].startswith("<b>") else "") + ">" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw {cls}"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def chart():
    """Two series over the main line of versions: label chrF and real-name chrF. One y scale (chrF 0-100, shown 30-100)."""
    pts = [("Fast", 42.6, 39.1), ("v3", 43.2, 33.6), ("v4 C", 51.0, 38.4), ("v5 MAX", 49.1, 45.7), ("v6", 79.6, 66.9), ("v7", 80.8, 67.2), ("v8", 83.1, 67.7), ("v9", 86.0, 67.2), ("v10", 86.8, 67.3), ("v11", 87.6, 67.6), ("v14", 87.7, 67.5)]
    Wd, Ht, L, R, Tp, Bt = 780, 340, 44, 180, 18, 40; lo, hi = 30, 100
    x = lambda i: L + i * (Wd - L - R) / (len(pts) - 1); y = lambda v: Tp + (hi - v) * (Ht - Tp - Bt) / (hi - lo)
    g = ""
    for v in (40, 60, 80, 100):
        g += f'<line x1="{L}" x2="{Wd - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/><text x="{L - 8}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{v}</text>'
    g += f'<line x1="{L}" x2="{Wd - R}" y1="{y(95.1):.1f}" y2="{y(95.1):.1f}" class="ref"/><text x="{Wd - R + 8}" y="{y(95.1) + 4:.1f}" class="lab">Hy-MT2-7B, labels: 95.1</text>'
    for i, (n, a, b) in enumerate(pts):
        g += f'<text x="{x(i):.1f}" y="{Ht - Bt + 18}" class="ax" text-anchor="middle">{E(n)}</text>'
    for k, cls in ((1, "s1"), (2, "s2")):
        g += f'<polyline class="ln {cls}" points="' + " ".join(f"{x(i):.1f},{y(p[k]):.1f}" for i, p in enumerate(pts)) + '"/>'
    for i, (n, a, b) in enumerate(pts):
        g += f'<circle class="pt s1f" cx="{x(i):.1f}" cy="{y(a):.1f}" r="4"/><circle class="pt s2f" cx="{x(i):.1f}" cy="{y(b):.1f}" r="4"/>'
        g += f'<rect class="hit" x="{x(i) - 26:.1f}" y="{Tp}" width="52" height="{Ht - Tp - Bt}" data-n="{E(n)}" data-a="{a}" data-b="{b}"/>'
    g += f'<text x="{Wd - R + 8}" y="{y(87.7) + 4:.1f}" class="lab">Labels, 87.7</text><text x="{Wd - R + 8}" y="{y(67.5) + 4:.1f}" class="lab">Real product names, 67.5</text>'
    rows = "".join(f"<tr><td>{E(n)}</td><td>{a}</td><td>{b}</td></tr>" for n, a, b in pts)
    return (f'<figure class="viz"><figcaption><b>Accuracy by version</b><span>chrF, 0–100, higher is better. The jump at v6 is the switch to a pretrained model.</span></figcaption>'
            f'<div class="legend"><span><i class="sw s1b"></i>Short labels (320 hand-written)</span><span><i class="sw s2b"></i>Real product names (993)</span><span><i class="sw refb"></i>Hy-MT2-7B, labels</span></div>'
            f'<div class="tw"><svg viewBox="0 0 {Wd} {Ht}" role="img" aria-label="Line chart of label and real-name chrF by version" id="chart">{g}</svg></div><div id="tip" class="tip" hidden></div>'
            f'<details><summary>Show the numbers</summary>{table(["Version", "Labels chrF", "Real names chrF"], [[E(n), a, b] for n, a, b in pts])}</details></figure>')


def vs_hy(X):
    """v14 against Hy-MT2-1.8B and 7B: grouped bars for accuracy (one chrF scale) and log-scale bars for size and time."""
    S3 = [("v14", "s1"), ("Hy-MT2-1.8B", "s2"), ("Hy-MT2-7B", "s3")]
    groups = [("Short labels", "320 hand-written", X["labels"]), ("Mixed English + Turkish", "64 labels", X["mixed"]),
              ("Real product names", "same 330 names for all three", X["names"]), ("Labels v14 marks reliable", "217 of the 320", X["reliable"])]
    Wd, Ht, L, Tp, Bt = 780, 330, 40, 26, 56; lo, hi = 0, 100; gw = (Wd - L - 10) / len(groups); bw = 30; gap = 4
    y = lambda v: Tp + (hi - v) * (Ht - Tp - Bt) / (hi - lo); g = ""
    for v in (0, 25, 50, 75, 100):
        g += f'<line x1="{L}" x2="{Wd - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/><text x="{L - 8}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{v}</text>'
    for gi, (name, sub, vals) in enumerate(groups):
        cx = L + gw * gi + gw / 2; x0 = cx - (3 * bw + 2 * gap) / 2
        for si, ((mn, cls), v) in enumerate(zip(S3, vals)):
            x = x0 + si * (bw + gap); top = y(v); h = y(0) - top
            g += (f'<path class="bar {cls}f" d="M{x:.1f},{y(0):.1f} V{top + 4:.1f} Q{x:.1f},{top:.1f} {x + 4:.1f},{top:.1f} H{x + bw - 4:.1f} Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + 4:.1f} V{y(0):.1f} Z"/>'
                  f'<text x="{x + bw / 2:.1f}" y="{top - 6:.1f}" class="val" text-anchor="middle">{v:.1f}</text>'
                  f'<rect class="hit" x="{x - 2:.1f}" y="{Tp}" width="{bw + 4}" height="{Ht - Tp - Bt}" data-tip="&lt;b&gt;{E(mn)}&lt;/b&gt;&lt;br&gt;{E(name)}: {v:.1f} chrF"/>')
        g += f'<text x="{cx:.1f}" y="{Ht - Bt + 20}" class="lab" text-anchor="middle">{E(name)}</text><text x="{cx:.1f}" y="{Ht - Bt + 36}" class="ax" text-anchor="middle">{E(sub)}</text>'
    leg = "".join(f'<span><i class="sq {c}b"></i>{E(n)}</span>' for n, c in S3)
    acc = (f'<figure class="viz"><figcaption><b>Accuracy: v14 against Hy-MT2</b><span>chrF, 0–100, higher is better. Each model alone on the same items; v14 as the full system.</span></figcaption>'
           f'<div class="legend">{leg}</div><div class="tw"><svg viewBox="0 0 {Wd} {Ht}" role="img" aria-label="Grouped bar chart: v14, Hy-MT2-1.8B and Hy-MT2-7B on four tests" class="hov">{g}</svg></div>'
           f'<details><summary>Show the numbers</summary>{table(["Test", "v14", "Hy-MT2-1.8B", "Hy-MT2-7B"], [[E(n + " (" + sb + ")"), *[f"{v:.1f}" for v in vals]] for n, sb, vals in groups])}</details></figure>')
    # cost: log scale, 1 to 10,000
    import math
    rows = [("Parameters (millions)", "M", [56.5, 1800, 7000], ["0.056B (56.5 million)", "1.8B", "7B"]),
            ("Size of the model files", "MB", [57, 1133, 4625], ["57 MB", "1,133 MB", "4,625 MB"]),
            ("Time per short label", "ms", [6, 600, 1400], ["about 6 ms, 1 thread", "400–800 ms, 4 threads", "about 1,400 ms, 4 threads"])]
    Wd2, L2, R2 = 780, 120, 300; lx = lambda v: L2 + math.log10(v) / 4 * (Wd2 - L2 - R2); rh = 22; Ht2 = 40 + len(rows) * (3 * (rh + 6) + 34); g = ""; yy = 18
    for t in (1, 10, 100, 1000, 10000):
        g += f'<line x1="{lx(t):.1f}" x2="{lx(t):.1f}" y1="34" y2="{Ht2 - 26}" class="grid"/><text x="{lx(t):.1f}" y="{Ht2 - 8}" class="ax" text-anchor="middle">{t:,}</text>'
    for title, unit, vals, labs in rows:
        g += f'<text x="0" y="{yy + 12}" class="lab"><tspan font-weight="600">{E(title)}</tspan></text>'; yy += 22
        for (mn, cls), v, lb in zip(S3, vals, labs):
            mult = "" if mn == "v14" else f" · {'about ' if unit == 'ms' else ''}{v / vals[0]:,.0f}× v14"
            g += (f'<text x="{L2 - 10}" y="{yy + 15}" class="ax" text-anchor="end">{E(mn)}</text>'
                  f'<path class="bar {cls}f" d="M{L2},{yy} H{lx(v) - 4:.1f} Q{lx(v):.1f},{yy} {lx(v):.1f},{yy + 4} V{yy + rh - 4} Q{lx(v):.1f},{yy + rh} {lx(v) - 4:.1f},{yy + rh} H{L2} Z"/>'
                  f'<text x="{lx(v) + 8:.1f}" y="{yy + 15}" class="val">{E(lb + mult)}</text>'
                  f'<rect class="hit" x="{L2}" y="{yy - 2}" width="{Wd2 - L2 - R2}" height="{rh + 4}" data-tip="&lt;b&gt;{E(mn)}&lt;/b&gt;&lt;br&gt;{E(title)}: {E(lb)}"/>')
            yy += rh + 6
        yy += 12
    cost = (f'<figure class="viz"><figcaption><b>Cost: parameters, size and speed</b><span>Log scale: each grid line is ten times the one before. Lower is better. Hy-MT2 was given 4 CPU threads, v14 one, so the real speed gap is larger than shown.</span></figcaption>'
            f'<div class="legend">{leg}</div><div class="tw"><svg viewBox="0 0 {Wd2} {Ht2}" role="img" aria-label="Log-scale bars of model size and time per label" class="hov">{g}</svg></div></figure>')
    return acc, cost


def bars(title, note, items, unit=""):
    mx = max(v for _, v, _ in items)
    r = "".join(f'<div class="brow"><span class="bn">{n}</span><span class="bt"><i class="{"bf on" if hl else "bf"}" style="width:{v / mx * 100:.1f}%"></i></span><span class="bv">{v}{unit}</span></div>' for n, v, hl in items)
    return f'<figure class="viz"><figcaption><b>{title}</b><span>{note}</span></figcaption><div class="bars">{r}</div></figure>'


def main():
    P1b = load(os.path.join(W, "places_v12_before.json"), {"total": [109, 160]}); P1a = load(os.path.join(W, "places_v13.json"), {"total": [148, 160]}); P1c = load(os.path.join(W, "places_v14.json"), {"total": [153, 160]})
    P2a = load(os.path.join(W, "places_2_v13.json"), {"total": [75, 101]}); P2c = load(os.path.join(W, "places_2_v14.json"), {"total": [91, 101]})
    IX = load(os.path.join(ROOT, "data", "gazetteer", "index.json"), {"meta": {}, "mb": {}}); gm = IX["meta"]; gmb = sum(IX["mb"].values())
    RL = load(os.path.join(W, "reliable_student_V14.json"), {"test": {"reliable_share": 0.678, "reliable_exact": 0.885, "reliable_chrF": 95.9, "rest_chrF": 70.5}})["test"]
    RV = load(os.path.join(W, "v12_vs_hy7_reliable.json"), {"reliable": {"hy7": [91.2, 95.3]}, "needs review": {"v11": [50.5, 70.7], "hy7": [88.3, 94.7]}})
    MX = load(os.path.join(W, "v12_mixed_table.json"), {}); D14 = load(os.path.join(W, "diff_student_V14.json"), {})
    mt = lambda rows: f'{sum(r["exact"] for r in rows) / len(rows) * 100:.1f}% / {sum(r["chrf"] for r in rows) / len(rows):.1f}' if rows else "-"
    f2 = lambda x: f"{x[0]:.1f}% / {x[1]:.1f}"
    S = []

    # ---------- summary
    S.append(f'''<header><p class="eyebrow">Project report · Nov 4 – Nov 15, 2026</p><h1>Label Normalizer</h1>
<p class="lede">An offline system that turns product names, item labels and addresses written in 16 languages into English, on a phone or a server. This page covers everything from the first version to v14: what was built, how each model was made, what was measured, what failed, and where every file is.</p>
<div class="kpis">
<div><b>76.2%</b><span>of 320 hand-written labels exactly right (87.7 chrF). Rules alone: 20.0%.</span></div>
<div><b>{P1c["total"][0]} / {P1c["total"][1]}</b><span>places, addresses and transliterations right on the first places test.</span></div>
<div><b>0.056B</b><span>parameters (56.5 million, 57 MB), about 6 ms per label on one CPU thread. Hy-MT2: 1.8B and 7B.</span></div>
<div><b>23</b><span>models and versions built in about 275 hours, on CPU only.</span></div>
</div></header>
<nav aria-label="Contents"><a href="#result">Result</a><a href="#how">How it works</a><a href="#scores">Scores</a><a href="#vshy">v14 vs Hy-MT2</a><a href="#base">Against base models</a><a href="#versions">Every version</a><a href="#story">How it was made</a><a href="#techniques">Techniques</a><a href="#failed">What failed</a><a href="#lessons">Lessons</a><a href="#limits">Limits</a><a href="#use">Using it</a><a href="#data">Data and licenses</a><a href="#time">Time</a><a href="#links">All links</a></nav>''')

    # ---------- result
    S.append(f'''<section id="result"><h2>The result</h2>
<p>Two models are final. <b>v14</b> is the one to use: the most accurate, 0.056B parameters (57 MB), about 6 ms per short label. <b>v8 Fast</b> is a smaller, faster cut (0.042B parameters, 42 MB, about 3 ms) meant for quick validation runs. Both were trained only with teachers and data that allow commercial use, with attribution.</p>
<p>The task is narrow on purpose. Inputs are short: a product label ("кроссовки для бега"), a marketplace product name, an address, a city. The output is the English a catalogue would use ("running shoes"), or the romanized form when transliteration is asked for.</p>
{chart()}
<div class="note"><b>How to read the scores.</b> <i>Exact</i> is the share of outputs equal to an accepted English answer after lowercasing. <i>chrF</i> (0–100) gives partial credit by character overlap, so "cotton tee shirt" for "cotton t-shirt" scores high but not 100. The two main test sets were never used for training: 320 product labels written by hand in 16 languages, and 993 real marketplace product names held out before training. Later tests (mixed-language labels, places) were each written before the work they measure.</div>
</section>''')

    # ---------- how it works
    S.append('''<section id="how"><h2>How the system works</h2>
<p>A label goes through four steps. The first three are rules and take well under a millisecond; only the fourth runs the neural model.</p>
<ol class="flow">
<li><b>English check.</b> A 100 KB character model says whether the label is already English. If it is clearly English it is returned as it is. Since v12 each word is also checked against a list of 26,000 English words, so "led ampul" is not mistaken for English.</li>
<li><b>Places and addresses.</b> The place list (2.1 million names of countries, regions, cities, towns, villages, islands and districts) gives the English name. Address rules translate the street-type word and romanize the street name: "ул. Ленина 5" becomes "Lenina Street 5".</li>
<li><b>Known words.</b> A product word list (about 1,800 terms per language) translates labels in which every word is known.</li>
<li><b>Neural translator.</b> Everything else goes to a 57 MB translation model with a small expert for each language. Its answer is used when its confidence is high enough; otherwise the romanized form is kept.</li>
</ol>
<p>Every answer carries a <b>reliable</b> or <b>needs review</b> mark. When transliteration is asked for instead of translation, only romanization rules run, with tables for Korean, Chinese and Japanese personal names and separate rules for Ukrainian, Greek, Arabic and Hindi.</p>
<p>The same logic exists twice, in Python and in JavaScript, and the two are compared on 3,018 inputs after every change. They give identical answers on all rule steps; the neural step differs on fewer than 1% of labels, where two candidate answers are nearly tied.</p>
</section>''')

    # ---------- scores
    places = table(["Places test", "v12", "v13", "v14"],
                   [["First test (160 items: countries, cities, towns, regions, addresses, transliteration)", f'{P1b["total"][0]}', f'{P1a["total"][0]}', f'<b>{P1c["total"][0]}</b>'],
                    ["Second test (101 items: islands, districts, villages, abbreviated addresses, names)", "not measured", f'{P2a["total"][0]}', f'<b>{P2c["total"][0]}</b>']])
    mixed = table(["Model", "English + Turkish test (64 labels): exact / chrF", "Size"],
                  [["<b>v14</b>", f'<b>{mt(D14.get("mixed_tr.json"))}</b>', "57 MB"]] +
                  ([["v11", f2(MX["v11"]["mixed_tr.json"]), "57 MB"], ["v10", f2(MX["v10"]["mixed_tr.json"]), "57 MB"], ["Hy-MT2-1.8B", f2(MX["hy18"]["mixed_tr.json"]), "1,133 MB"], ["Hy-MT2-7B", f2(MX["hy7"]["mixed_tr.json"]), "4,625 MB"]] if MX else []))
    rel = table(["Test labels", "Share", "v14: exact / chrF", "Hy-MT2-7B on the same labels"],
                [["Marked reliable", f'{RL["reliable_share"] * 100:.0f}%', f'<b>{RL["reliable_exact"] * 100:.1f}% / {RL["reliable_chrF"]:.1f}</b>', f'{RV["reliable"]["hy7"][0]}% / {RV["reliable"]["hy7"][1]}'],
                 ["Marked needs review", f'{100 - RL["reliable_share"] * 100:.0f}%', f'{RV["needs review"]["v11"][0]}% / {RL["rest_chrF"]:.1f}', f'{RV["needs review"]["hy7"][0]}% / {RV["needs review"]["hy7"][1]}']])
    S.append(f'''<section id="scores"><h2>Scores of the final system</h2>
{table(["Test", "What it holds", "v14", "v8 Fast"],
       [["Short labels", "320 hand-written product labels, 16 languages", "<b>76.2% exact / 87.7 chrF</b>", "65.0% / 79.7"],
        ["Short labels never seen in training", "the 260 of those whose text is not in the training data", "<b>73.1% / 86.1</b>", "59.6% / 76.7"],
        ["Real product names", "993 marketplace names written by people", "<b>67.5 chrF</b>", "62.2 chrF"],
        ["Address cases", "70 tuned and 48 unseen addresses and places", "<b>70 of 70 · 45 of 48</b>", "70 of 70 · 44 of 48"],
        ["Time per request", "one CPU thread, 8-bit model", "<b>about 6 ms per label, 24 ms per long name</b>", "about 3 ms, 17 ms"]])}
<h3>Labels that mix two languages</h3>
<p>A label that is English except for one word, or the other way round, must still be translated ("black deri ceket", "led ampul"). The test was written by hand in English + Turkish and never trained on.</p>
{mixed}
<h3>Places, addresses and transliteration</h3>
<p>Both tests were written by hand before the work they measure. One accepted answer was added to the first test afterwards because it is equally correct ("Osaka City"); all columns are scored the same way.</p>
{places}
<h3>The reliable mark</h3>
<p>An answer is marked reliable when it is a known place, clearly English text, made only of known words, or a neural answer with confidence of at least 0.82. The threshold was set on separate tuning labels.</p>
{rel}
<p>On the two thirds it marks reliable, v14 is level with the 7-billion-parameter model in chrF. The mark does not remove errors: about one reliable answer in nine still differs from the reference, usually by a near-synonym or a plural.</p>
</section>''')

    # ---------- base
    S.append(f'''<section id="base"><h2>Against the base models</h2>
<p>Base models are existing translators, run alone on the same tests and the same machine. Ours are run as the full system. Parameters are given in billions (B), the way translation models are usually named: v14 has 56.5 million, which is 0.056B; v8 Fast has 0.042B.</p>
{table(["Model", "Labels: exact / chrF", "Real names: chrF", "Parameters", "Size", "Time per label / per product name"],
       [["<b>v14</b>", "<b>76.2% / 87.7</b>", "<b>67.5</b>", "<b>0.056B</b>", "<b>57 MB</b>", "<b>6 ms / 24 ms</b> (1 thread)"],
        ["v8 Fast", "65.0% / 79.7", "62.2", "0.042B", "42 MB", "3 ms / 17 ms (1 thread)"],
        ["Hy-MT2-7B (Tencent)", "90.3% / 95.1", "57.5 (330-name sample)", "7B", "4,625 MB", "about 1,400 ms / 6,000 ms (4 threads)"],
        ["Hy-MT2-1.8B (Tencent)", "60.6% / 79.1", "51.9", "1.8B", "1,133 MB", "400–800 ms / 1,550 ms (4 threads)"],
        ["OPUS-MT, one model per language", "52.8% / 76.2", "47.1", "about 0.077B each", "about 300 MB each", "not timed / 289 ms"],
        ["NLLB-200 600M (Meta, non-commercial)", "51.9% / 70.9", "48.7", "0.6B", "870 MB", "not timed / 748 ms"],
        ["OPUS-MT mul-en, before our training", "10.3% / 34.2", "38.4", "0.077B", "about 300 MB", "not timed"]], "wide")}
{bars("Short labels, chrF", "The base model v6–v14 start from is the last bar.", [("Hy-MT2-7B · 7B", 95.1, False), ("v14 · 0.056B", 87.7, True), ("Hy-MT2-1.8B · 1.8B", 79.1, False), ("OPUS-MT per language · 300 MB each", 76.2, False), ("NLLB-200 600M · 870 MB", 70.9, False), ("OPUS-MT mul-en, untrained · 300 MB", 34.2, False)])}
<p><b>v14 beats every base model except Hy-MT2-7B on short labels, and beats all of them on real product names.</b> Hy-MT2-7B stays ahead on short labels by 7.4 chrF and on mixed English + Turkish labels by about 4: a model 80 times larger knows more rare product words. v14 answers about 200 times faster.</p>
</section>''')

    # ---------- v14 against Hy-MT2
    XX = load(os.path.join(W, "v14_vs_hy_extra.json"), {}); H18 = load(os.path.join(W, "teachers", "mixed_hy18.json"), {}); H7 = load(os.path.join(W, "teachers", "mixed_hy7.json"), {})
    A330 = load(os.path.join(W, "teachers", "abo_sample330.json"), {}); G18 = load(os.path.join(W, "teachers", "gold_hy18.json"), {}); G7 = load(os.path.join(W, "teachers", "gold_hy7.json"), {})
    avg = lambda rows: sum(r["chrf"] for r in rows) / len(rows)
    X = {"labels": [87.7, G18["chrF"], G7["chrF"]], "mixed": [avg(D14["mixed_tr.json"]), avg(H18["mixed_tr.json"]), avg(H7["mixed_tr.json"])],
         "names": [XX["v14_abo330"]["chrF"], A330["hy18"]["chrF"], A330["hy7"]["chrF"]], "reliable": [RL["reliable_chrF"], XX["hy18_reliable"][1], RV["reliable"]["hy7"][1]]}
    acc, cost = vs_hy(X)
    win = lambda a, b: "<b>v14</b>" if a > b + 1 else ("Hy-MT2-7B" if b > a + 1 else "level")
    S.append(f'''<section id="vshy"><h2>Is v14 better than Hy-MT2?</h2>
<p><b>Better than Hy-MT2-1.8B on every test. Better than Hy-MT2-7B on real product names, size and speed, but not on short labels.</b> Hy-MT2 is Tencent's translation model; the 7-billion-parameter version is the most accurate model available to this project, and it was also one of the teachers.</p>
{acc}
{cost}
{table(["Question", "v14", "Hy-MT2-1.8B", "Hy-MT2-7B", "Better"],
       [["Short labels: exact / chrF", f"76.2% / {X['labels'][0]:.1f}", f"{G18['exact'] * 100:.1f}% / {X['labels'][1]:.1f}", f"{G7['exact'] * 100:.1f}% / {X['labels'][2]:.1f}", win(X['labels'][0], X['labels'][2])],
        ["Mixed English + Turkish: chrF", f"{X['mixed'][0]:.1f}", f"{X['mixed'][1]:.1f}", f"{X['mixed'][2]:.1f}", win(X['mixed'][0], X['mixed'][2])],
        ["Real product names, same 330: chrF", f"{X['names'][0]:.1f}", f"{X['names'][1]:.1f}", f"{X['names'][2]:.1f}", win(X['names'][0], X['names'][2])],
        ["Labels v14 marks reliable: chrF", f"{X['reliable'][0]:.1f}", f"{X['reliable'][1]:.1f}", f"{X['reliable'][2]:.1f}", win(X['reliable'][0], X['reliable'][2])],
        ["Parameters", "0.056B", "1.8B", "7B", "<b>v14</b>, 124× fewer than the 7B"],
        ["Size", "57 MB", "1,133 MB", "4,625 MB", "<b>v14</b>, 81× smaller"],
        ["Time per short label", "about 6 ms", "400–800 ms", "about 1,400 ms", "<b>v14</b>, about 230× faster"],
        ["Runs on a phone, offline", "yes", "no", "no", "<b>v14</b>"],
        ["Places, addresses, transliteration", "153 of 160 · 91 of 101", "not run", "not run", "not compared"]], "wide")}
<ul>
<li><b>Where Hy-MT2-7B wins:</b> short labels by {X['labels'][2] - X['labels'][0]:.1f} chrF and mixed-language labels by {X['mixed'][2] - X['mixed'][0]:.1f}. A model 80 times larger knows more rare product words in every language.</li>
<li><b>Where v14 wins:</b> real marketplace product names by {X['names'][0] - X['names'][2]:.1f} chrF on the same 330 names, because v14 learned the catalogue's naming style from 35,000 real pairs; and size, speed and running offline on a phone.</li>
<li><b>On the two thirds of labels v14 marks reliable</b> the two are level in chrF ({X['reliable'][0]:.1f} against {X['reliable'][2]:.1f}); the 7B still has more exact matches there ({RL['reliable_exact'] * 100:.1f}% against {RV['reliable']['hy7'][0]}%).</li>
<li><b>Choose v14</b> for phones, offline use and high volume; send the labels it marks <i>needs review</i> to a person or to the 7B when the best possible answer matters more than time.</li>
</ul>
<p>Real-name scores here use the 330-name sample on which Hy-MT2-7B was run (it needs about 6 seconds per name), so all three are scored on exactly the same names; on all 993 names v14 scores 67.5. The places tests were not run on Hy-MT2.</p>
</section>''')

    # ---------- every version
    rows = []
    for n, k, d, ex, ch, ab, mb, l, nm, com, idea in M:
        nb = f"<b>{E(n)}</b>" if k in ("V14", "V8F") else E(n)
        rows.append([nb, d, f"{ex:.1f}%", f"{ch:.1f}", f"{ab:.1f}", mb, l, nm, "yes" if com else "no"])
    S.append(f'''<section id="versions"><h2>Every version, one table</h2>
<p>Full system on the same two tests. Sizes are the 8-bit model files. Times are medians on one CPU thread; values with "~" are averages from that model's own test run, and v14 was timed side by side with v13 and found equal. "Commercial" says whether the teachers and data behind the model allow commercial use.</p>
{table(["Model", "Made", "Labels: exact", "Labels: chrF", "Real names: chrF", "Size (MB)", "Label (ms)", "Name (ms)", "Commercial"], rows, "wide")}
<p>v12, v13 and v14 use exactly v11's neural weights. What changed in them is the rules and data around the model, which is why their product scores are equal and their place scores are not.</p>
</section>''')

    # ---------- story
    story = [
        ("Rules first (Nov 4–5)", "The English check and Fast were built before any neural model: a place list from GeoNames and Unicode CLDR, romanization rules for nine writing systems, address rules, and a hand-written product word list. Python and JavaScript versions were kept identical from the first day. Fast alone scores 42.6 chrF on labels and handles addresses in about 0.3 ms."),
        ("Small models from nothing: v3, v4, v5 (Nov 5–10)", "Twelve character-level models of 7 to 22 million parameters were trained from random weights on product titles translated into the 16 languages by larger models (the teachers). More data, longer training, a second and third teacher, and the teachers' own answers as extra targets moved label chrF from 43 to 51. Adding 35,000 real product names written by people (v5 MAX) lifted real-name chrF by 7 points, the largest single gain of this phase."),
        ("JEPA experiment (Nov 10–11)", "A Joint-Embedding Predictive Architecture was built and tuned: the encoder learns by predicting the meaning vector of a label, then a translator is trained on top. It worked technically (the right English name out of 1,000 was found 19% of the time by vector alone) but a baseline given the same hours of ordinary training scored higher (45.9 against 43.5)."),
        ("The pretrained start: v6 (Nov 11–12)", "A review showed the weak point: the neural part scored about 32 chrF alone, while large translators score 76 to 95. v6 starts from a pretrained translator (OPUS-MT mul-en, 77M parameters, 34 chrF untrained on this task), with its vocabulary cut to our languages and 3,091 missing pieces added, mostly Korean. Fine-tuning for 8.5 hours on all product data plus 58,000 short labels made from real search queries gave 79.6 on labels and 66.9 on real names. The 8-bit conversion needed per-channel quantization to keep that accuracy."),
        ("Experts per language: v7 (Nov 12)", "v6 was frozen and a small pair of matrices per language was added to every layer, routed by the language detector, so only the detected language's parameters are active. Gain: 1.2 chrF."),
        ("Smaller and faster: v8 (Nov 13)", "Profiling showed most time went to the decoder and the output layer. v8 keeps v6's 6-layer encoder but only 2 decoder layers, an output layer with English pieces only, and experts in the encoder only. Result: half the size, twice the speed, and 3.5 chrF better on labels. v8 Fast is cut further from it."),
        ("A tool for adding data: v9 and v10 (Nov 13–14)", "The post-training tool takes a parent model and new pairs, trains with the old data replayed, exports, retunes the rules and prints PASS or FAIL against the parent. v9 used it with 200,000 more short labels; v10 added beam search and common terms written in all 16 languages. Labels reached 86.8 chrF."),
        ("Cleaning the data with a judge: v11 (Nov 14)", "v10 was run over 28,493 training labels and missed 10,685. Hy-MT2-7B re-translated 8,331 of them. In about seven of ten cases the training answer itself was wrong or poor, not the model. 5,976 bad pairs were removed and 5,921 corrected ones taught. v11 also added the English double check (9 of 320 test labels had been returned untranslated) and the reliable mark."),
        ("One foreign word: v12 (Nov 15)", "A new English + Turkish test showed labels like \"led ampul\" coming back untranslated. Training on 44,000 mixed-language pairs made ordinary labels worse and was rejected, as were blends of old and new weights. The neural model alone already scored 90.6 on the test; the loss was in the rules. A word-level English check fixed it without touching the weights."),
        ("Places of the whole world: v13 and v14 (Nov 15)", f"The place list was rebuilt twice from GeoNames: first every town of 500 people or more with regions and preferred English names, then villages from 200 people, islands, districts, neighbourhoods and every place with a local-script name ({gm.get('entries', 0):,} names, {gmb:.0f} MB). Address rules were extended (postal codes, German street names, abbreviations across languages, Korean and Japanese forms), personal-name tables and Ukrainian and Greek rules were added, and about 7,500 Arabic word readings were mined from place names because Arabic is written without short vowels. The two places tests went from 109 to 153 of 160 and from 75 to 91 of 101."),
    ]
    S.append('<section id="story"><h2>How it was made, in order</h2><div class="story">' + "".join(f"<div><h3>{E(t)}</h3><p>{E(b)}</p></div>" for t, b in story) + "</div></section>")

    # ---------- techniques
    tech = [
        ("Data", [["Back-translation", "English product titles and search queries translated into each language by a teacher model, kept only when they translate back to the original."],
                  ["Real paired data", "35,000 marketplace names written by people in 12 languages with an English name."],
                  ["Short-label data", "Search queries as stand-ins for labels, since the tests are short and titles are long."],
                  ["Teacher as judge", "A stronger model re-translates what the student gets wrong, separating model errors from bad training answers."],
                  ["Held-out discipline", "Queries equal to a test answer are removed from training; each new test is written before the work it measures."]]),
        ("Model", [["Pretrained start", "Fine-tuning a 77M-parameter translator instead of training from nothing."],
                   ["Vocabulary surgery", "Unused word pieces removed, missing ones added, separate source and English-only output vocabularies."],
                   ["Shallow decoder", "6 encoder layers, 2 decoder layers: the encoder runs once, the decoder once per output piece."],
                   ["Language experts", "A 1.6 MB adapter per language in the encoder, selected by the language detector."],
                   ["Post-training with replay", "New pairs sampled six times more often while all old data is replayed, with an automatic pass/fail gate."]]),
        ("After training", [["Per-channel 8-bit conversion", "Recovers about 3 chrF that the default conversion loses."],
                            ["Beam search", "Two candidates kept for short labels; chosen on tuning data."],
                            ["Head switch-off tests", "Each attention head disabled in turn on the small models; a change kept only if two held-out sets agree."],
                            ["Checkpoint averaging and weight blends", "Tried on v4 and v12; kept only where tuning data improved."]]),
        ("Rules around the model", [["Decision rule", "Places, addresses and known words stay with the rules; the neural answer needs a minimum confidence. Tuned on separate labels."],
                                    ["English double check and word check", "A second opinion when the English check is unsure, and a per-word check against an English word list."],
                                    ["Unknown-word guard", "One unknown word and a hesitant translator: keep the romanized form."],
                                    ["Reliable mark", "A calibrated confidence threshold separates answers to trust from answers to check."],
                                    ["Mined readings", "Arabic and Hindi word readings taken from aligned place names."]]),
    ]
    S.append('<section id="techniques"><h2>Techniques used</h2>' + "".join(f"<h3>{E(t)}</h3>" + table(["Technique", "What it does here"], [[f"<b>{E(a)}</b>", E(b)] for a, b in r]) for t, r in tech) + "</section>")

    # ---------- failed
    S.append(f'''<section id="failed"><h2>What was tried and rejected</h2>
{table(["Attempt", "What happened", "Decision"],
       [["Word pieces (BPE) and a 52M model, from nothing", "Learned too slowly on 4 CPU cores to finish.", "Dropped; stayed with characters until v6."],
        ["Hy-MT2 as the only teacher (v5 HY)", "Its answers are the best, but it is slow, so only 8k pairs: 45.5 chrF.", "A good teacher with little data loses to an average teacher with a lot."],
        ["JEPA pretraining", "Baseline with the same hours scored 2.4 chrF higher.", "Kept as a documented experiment, not used."],
        ["JEPA nearest-name lookup", "Never beat the rules.", "Dropped."],
        ["Default 8-bit conversion of v6", "Lost 5 chrF on short labels.", "Replaced by per-channel conversion."],
        ["Fine-tuning after switching heads off (v3)", "Hurt the held-out labels.", "Rejected; heads are switched off without retraining."],
        ["v11, first run: mixed labels made by joining two labels", "Taught the model to keep foreign word order.", "Discarded; rebuilt by swapping one known word."],
        ["v12 training on 44,000 mixed-language pairs", "Tuning labels fell from 90.3 to 88.3 chrF; the gate said FAIL.", "Rejected; the weights stayed v11's."],
        ["Blending v11 and v12 weights", "No blend beat v11's weights on tuning data.", "Rejected."],
        ["First rebuilt place list (v13)", "Read the Russian word for hair as the Greek city Volos.", "Fixed with a product-word block list before release."],
        ["Cross-language street rule, first version (v14)", "Read Polish “al.” as Portuguese.", "Caught by the places test and fixed before release."]], "wide")}
</section>''')

    # ---------- lessons
    S.append('''<section id="lessons"><h2>What we learned</h2>
<ul class="lessons">
<li><b>The starting point mattered more than everything else combined.</b> Seven days and twelve from-scratch models moved label chrF from 43 to 51. One 8.5-hour fine-tune of a pretrained translator moved it to 80.</li>
<li><b>Data that looks like the task beats more data.</b> Real product names and short search queries each gave larger gains than doubling the translated titles.</li>
<li><b>Much of the remaining error was in the training data.</b> Seven of ten labels the model "missed" had a wrong or poor training answer.</li>
<li><b>Measure the model alone before retraining.</b> For mixed-language labels, more training hurt; the fault was a rule that never let the label reach the model.</li>
<li><b>A smaller model can be better and faster</b> when the cut follows a measurement: most decoder layers and the multilingual output layer were not earning their cost.</li>
<li><b>Rules stay necessary.</b> Places and addresses are where a small neural model is weakest and a list is near-perfect.</li>
<li><b>Write the test before the fix.</b> Every regression in the last four versions was caught by a test written beforehand or by the automatic gate, not by reading outputs.</li>
<li><b>A gate that can say FAIL is worth more than a model that looks good.</b> It rejected two training runs that would otherwise have shipped.</li>
</ul></section>''')

    # ---------- limits
    S.append(f'''<section id="limits"><h2>Limits</h2>
<ul>
<li><b>Not better than Hy-MT2-7B on short labels</b> (87.7 against 95.1 chrF) or on mixed English + Turkish labels (89.7 against 93.9). Closing that gap needs far more of its answers as training data than a CPU can produce.</li>
<li><b>Not error-free.</b> About one answer in four on short labels differs from the reference; many are near-synonyms, some are wrong. The reliable mark tells which to check.</li>
<li><b>No list of streets.</b> Street-type words are translated in many forms and languages, but street names themselves are romanized. A list of every street would be hundreds of gigabytes.</li>
<li><b>Places still missed:</b> 17 of the 261 items in the two places tests, mainly some Japanese readings, Thai address words written without spaces, places named in a third language, and Ukrainian text with no Ukrainian-only letters.</li>
<li><b>The place list is large:</b> {gmb:.0f} MB, and about 875 MB of memory in Python with every writing system loaded (the v13 list: 32 MB and about 480 MB). A phone loads only the writing systems it meets.</li>
<li><b>Korean and Thai</b> remain the weakest languages for product labels.</li>
<li><b>Test sets are small</b> (320 labels from 20 products, 64 mixed labels, 261 place items). Differences under about 1 chrF between neighbouring versions are within noise.</li>
<li><b>Automated checks on GitHub have never run</b> for this pull request: every run ends in a few seconds with no runner assigned, which is an account setting. All checks were run on the build machine.</li>
</ul></section>''')

    # ---------- use
    S.append(f'''<section id="use"><h2>Using it and adding to it</h2>
<p>Python: <code>NEURAL_DIR=student_V14 python -m runtime.cli in.json</code>. JavaScript: <code>createNormalizer({{ model: "accurate", baseUrl }})</code>, then <code>normalize(items)</code>. Each result has the output, the method used, a confidence, the detected language and the reliable mark.</p>
{table(["To add", "Do this", "Takes"],
       [["A few words or names", "Put them in a three-column file and run <code>build/add_terms.py</code>. It tests itself and undoes the change if a score drops.", "seconds"],
        ["New translated pairs", "<code>sh build/post_train.sh student_V14 student_V15 120 my_pairs.jsonl</code>. Trains, exports, retunes, compares with the parent, prints PASS or FAIL.", "1–5 hours on 4 CPU cores"],
        ["An English-only list", "<code>build/make_pairs.py</code> writes the other languages with Hy-MT2-1.8B, then post-train.", "about 1 second per pair, then as above"],
        ["Places", "<code>sh build/geonames_all.sh</code>, then <code>python build/build_gazetteer3.py</code>; <code>MINPOP</code> sets the smallest village.", "about 10 minutes"]])}
<p>The step-by-step guide is {A(B + "/docs/ADDING.md", "docs/ADDING.md")}.</p>
</section>''')

    # ---------- data
    S.append(f'''<section id="data"><h2>Data and licenses</h2>
{table(["Source", "Used for", "License"],
       [["OPUS-MT mul-en (Helsinki-NLP)", "starting weights of v6 to v14", "CC BY 4.0"],
        ["OPUS-MT per-language models, M2M100", "teacher translations", "CC BY 4.0 · MIT"],
        ["Hy-MT2-1.8B and 7B (Tencent)", "teacher for short labels; the 7B as judge and reference", "Apache-2.0"],
        ["NLLB-200 600M (Meta)", "teacher for v3, v4 A / C, v5 C2 / ALL3 only", "CC BY-NC 4.0 (those models are non-commercial)"],
        ["Amazon ESCI", "English product titles and search queries", "Apache-2.0"],
        ["Amazon Berkeley Objects", "35,000 real product names in 12 languages; 993 held out as a test", "CC BY 4.0"],
        ["GeoNames", "place list; Arabic and Hindi word readings", "CC BY 4.0, attribution required"],
        ["Unicode CLDR", "country names in every language", "Unicode License"],
        ["Sudachi dictionary", "Japanese readings", "Apache-2.0"]])}
<p>v6 and every later model, v5 MAX, v5 B2, v5 H, v5 HY and v4 B can be used commercially with attribution. Details: {A(B + "/LICENSES.md", "LICENSES.md")}.</p>
</section>''')

    # ---------- time
    S.append(f'''<section id="time"><h2>Time spent</h2>
{table(["What", "Time"],
       [["From the first message to this report", "Nov 4 to Nov 15: about 275 hours (11.5 days), including many hours lost to machine restarts"],
        ["Training the neural models", "about 105 hours on 4 CPU cores"],
        ["Teacher translations and the 7B judge", "about 31 hours"],
        ["Weight tests and edits, JEPA tuning", "about 30 hours"],
        ["Rules, place lists, tests, tools, reports", "the rest"]])}
<p>Everything ran on a 4-core cloud CPU with no GPU. The machine restarted often, so every long step was written to continue where it stopped.</p>
</section>''')

    # ---------- links
    top = table(["What", "Link"],
                [["This report", A("https://claude.ai/artifact/QKvzmgbPuUJYixzyVx7Xzk", "claude.ai/artifact/QKvzmgbPuUJYixzyVx7Xzk")],
                 ["Document: all models, how they were made, lessons", A("https://claude.ai/artifact/LqFBZNhhMArpGUZxN49dU7", "claude.ai/artifact/LqFBZNhhMArpGUZxN49dU7")],
                 ["Phone test page (v14 and v8 Fast run in the browser)", A("https://claude.ai/artifact/3je2spXyFcH1QicAoMfZWL", "claude.ai/artifact/3je2spXyFcH1QicAoMfZWL")],
                 ["Pull request (draft): all code, models and data", A(GH + "/pull/1", "translation-model/pull/1")],
                 ["Whole branch as one zip", A(f"{GH}/archive/refs/heads/{BR}.zip", f"archive/refs/heads/{BR}.zip")],
                 ["This report as PDF", A(B + "/reports/pdf/label-normalizer-report.pdf", "reports/pdf/label-normalizer-report.pdf")],
                 ["The earlier, longer report (kept for its detail tables)", A(B + "/reports/pdf/label-normalizer-report-v1.pdf", "reports/pdf/label-normalizer-report-v1.pdf")],
                 ["Rules: data and code", A(T + "/data", "data/") + " · " + A(T + "/runtime", "runtime/")],
                 ["Place list and its builder", A(T + "/data/gazetteer", "data/gazetteer/") + " · " + A(B + "/build/build_gazetteer3.py", "build/build_gazetteer3.py")],
                 ["Training data (compressed)", A(T + "/training_data", "training_data/")],
                 ["Build and training scripts", A(T + "/build", "build/")],
                 ["Tests", A(T + "/test", "test/")],
                 ["Result files of every version", A(T + "/reports", "reports/")]], "wide")
    mrows = [["Fast (rules only)", A(T + "/data", "data/") + " · " + A(T + "/runtime", "runtime/"), "-", A(T + "/reports", "reports/"), ZIP["fast"]]]
    for n, k, d, *_ in M[1:]:
        f = FOLDER[k]
        mrows.append([f"<b>{E(n)}</b>" if k in ("V14", "V8F") else E(n), A(f"{T}/models/{f}", "models/" + f), A(f"{T}/weights/{f}_fp32", f"weights/{f}_fp32") if k in FP32 else "-",
                      A(f"{T}/reports/{REPORTS[k]}", "reports/" + REPORTS[k]) if k in REPORTS else A(T + "/reports", "reports/"), ZIP.get(k, "-")])
    S.append(f'''<section id="links"><h2>All links, since the first version</h2>
<p>GitHub links need access to the private repository and point at the work branch of the draft pull request. The pages on claude.ai are private to their owner until shared.</p>
{top}
<h3>Every model</h3>
{table(["Model", "Model files (8-bit, as shipped)", "Full-precision weights", "Result files", "Zip sent in the chat"], mrows, "wide")}
<p>Files sent in the chat have no web address; the same content is in the repository folders. v12, v13 and v14 share v11's neural weights; their folders differ in <code>vocab.json</code> (rule settings and word lists), and v13 and v14 need the place list from the same commit.</p>
</section>''')

    page = open(os.path.join(HERE, "report_v2_shell.html"), encoding="utf-8").read().replace("{{BODY}}", "\n".join(S))
    out = os.path.join(ROOT, "web", "report", "index.html"); open(out, "w", encoding="utf-8").write(page); print("wrote", out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
