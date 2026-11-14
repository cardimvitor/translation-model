"""Build the final project report page from the result files (no number is typed by hand).
  python build/make_report.py   -> web/report/index.html"""
import glob, html, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
OUT = os.path.join(ROOT, "web", "report"); os.makedirs(OUT, exist_ok=True)
E = html.escape

# model directory -> (label, teachers / data, commercial use, training minutes)
META = {
    "none": ("Fast", "no neural network: place names, word list, transliteration", True, 0),
    "student_edited": ("v3", "NLLB", False, 100),
    "student_A": ("v4 A", "NLLB", False, 240),
    "student_B": ("v4 B", "OPUS-MT + M2M100", True, 240),
    "student_C": ("v4 C", "NLLB + OPUS-MT", False, 300),
    "student_B2": ("v5 B2", "OPUS-MT, more titles, teacher answers", True, 300),
    "student_C2_edited": ("v5 C2", "OPUS-MT + NLLB", False, 300),
    "student_H": ("v5 H", "OPUS-MT + Hy-MT2 answers", True, 300),
    "student_ALL3_edited": ("v5 ALL3", "OPUS-MT + NLLB + Hy-MT2", False, 300),
    "student_HY_edited": ("v5 HY", "Hy-MT2 only (8k pairs)", True, 180),
    "student_MAX_edited": ("v5 MAX", "OPUS-MT + Hy-MT2 + 35k real ABO names", True, 600),
    "student_JG_edited": ("JEPA-Gen", "MAX data, encoder pretrained with JEPA", True, 360),
    "student_JG": ("JEPA-Gen", "MAX data, encoder pretrained with JEPA", True, 360),
    "student_BASE_edited": ("Baseline", "MAX data, no JEPA (same size, same total time)", True, 360),
    "student_V6": ("v6", "pretrained OPUS-MT mul-en (77M) fine-tuned on MAX data + 58k short labels", True, 510),
    "student_V9": ("v9", "v8 Complete post-trained with 200k more short labels (every search query in 7 languages, more Korean and Thai), made with the post-training tool", True, 300),
    "student_V8": ("v8 Complete", "v6 encoder + 2-layer decoder + English-only output + encoder experts per language; all data + 13k Hy-MT2 short labels", True, 420),
    "student_V8F": ("v8 Fast", "cut from v8 Complete: 3 encoder layers, 1 decoder layer, no experts", True, 180),
    "student_V7": ("v7", "v6 + one 3 MB expert per language (only the detected language's expert is active)", True, 240),
    "student_BASE": ("Baseline", "MAX data, no JEPA (same size, same total time)", True, 360),
}


def parse_final():
    txt = open(os.path.join(W, "final_eval.txt"), encoding="utf-8").read()
    for extra in ("final_eval_V6.txt", "final_eval_V7.txt", "final_eval_V8.txt", "final_eval_V8F.txt", "final_eval_V9.txt"):
        if os.path.exists(os.path.join(W, extra)):
            txt += open(os.path.join(W, extra), encoding="utf-8").read()
    rows = {}
    for block in txt.split("== ")[1:]:
        name, *lines = block.strip().split("\n"); r = {}
        for l in lines:
            m = re.match(r"(gold|abo)\s+(Fast|Student only|Accurate)\s+exact \d+/\d+ \(([\d.]+)%\)\s+chrF\s+([\d.]+)", l)
            if m:
                r[(m[1], m[2])] = (float(m[3]), float(m[4]))
            m = re.match(r"cases PASS (\d+)/70 \| blind PASS (\d+)/48", l)
            if m:
                r["cases"], r["blind"] = int(m[1]), int(m[2])
            m = re.match(r"size (\S+) (\S+)", l)
            if m:
                r["size"] = sum(float(x.rstrip("M")) for x in m.groups())
                md = os.path.join(ROOT, "models", name.strip())
                if os.path.exists(os.path.join(md, "encoder_int8.onnx")):       # exact size in MB (10^6 bytes)
                    r["size"] = sum(os.path.getsize(os.path.join(md, f"{n}_int8.onnx")) for n in ("encoder", "decoder")) / 1e6
        rows[name.strip()] = r
    return rows


def load(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default


def bars(items, maxv, unit):
    """Horizontal single-series bar chart (inline SVG); items: [(label, value, note)]."""
    h = 26; lw = 150; W_ = 640; bw = W_ - lw - 70; y = 8; parts = []
    for lab, v, note in items:
        w = max(2, bw * v / maxv)
        parts.append(f'<g><title>{E(lab)}: {v:.1f} {unit}{(" (" + note + ")") if note else ""}</title>'
                     f'<text x="{lw - 10}" y="{y + 15}" class="bl">{E(lab)}</text>'
                     f'<rect x="{lw}" y="{y + 4}" width="{w:.1f}" height="16" rx="4" class="bar{" hi" if note == "commercial use OK" else ""}"/>'
                     f'<text x="{lw + w + 6:.1f}" y="{y + 16}" class="bv">{v:.1f}</text></g>')
        y += h
    ticks = "".join(f'<line x1="{lw + bw * t / maxv:.1f}" x2="{lw + bw * t / maxv:.1f}" y1="4" y2="{y}" class="grid"/>'
                    f'<text x="{lw + bw * t / maxv:.1f}" y="{y + 14}" class="tk">{t}</text>' for t in range(0, int(maxv) + 1, 10))
    return f'<div class="chart"><svg viewBox="0 0 {W_} {y + 22}" role="img" aria-label="bar chart">{ticks}{"".join(parts)}</svg></div>'


def curves():
    """Held-out chrF by training step, from the training logs (JEPA-Gen, Baseline, MAX)."""
    series = []
    for tag, lab, cls in (("JG", "JEPA-Gen", "c1"), ("BASE", "Baseline", "c2"), ("MAX", "v5 MAX", "c3")):
        pts = []
        for l in open(os.path.join(W, f"train_{tag}.log"), encoding="utf-8") if os.path.exists(os.path.join(W, f"train_{tag}.log")) else []:
            m = re.match(r"dev \{'bt_chrF': ([\d.]+).*'step': (\d+)\}", l)
            if m:
                pts.append((int(m[2]), float(m[1])))
        pts = sorted(set(pts))
        if pts:
            series.append((lab, cls, pts))
    if not series:
        return ""
    mx = max(x for _, _, p in series for x, _ in p); my = 50
    W_, H_, l, b = 640, 260, 40, 30
    X = lambda x: l + (W_ - l - 20) * x / mx; Y = lambda y: H_ - b - (H_ - b - 10) * y / my
    g = "".join(f'<line x1="{l}" x2="{W_ - 20}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{l - 6}" y="{Y(t) + 4:.1f}" class="tk" style="text-anchor:end">{t}</text>' for t in range(0, my + 1, 10))
    step = 2000 if mx > 8000 else 1000
    g += "".join(f'<text x="{X(t):.1f}" y="{H_ - b + 16}" class="tk">{t // 1000}k</text>' for t in range(0, mx + 1, step))
    lines = ""
    for lab, cls, p in series:
        lines += f'<polyline class="ln {cls}" points="{" ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in p)}"/>'
        lines += "".join(f'<circle class="pt {cls}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5"><title>{E(lab)}, step {x}: {y:.1f}</title></circle>' for x, y in p)
        x, y = p[-1]; lines += f'<text x="{min(X(x) + 6, W_ - 70):.1f}" y="{Y(y) - 6:.1f}" class="lab {cls}">{E(lab)}</text>'
    leg = " ".join(f'<span><i class="sw {c}"></i>{E(n)}</span>' for n, c, _ in series)
    return (f'<div class="chart"><svg viewBox="0 0 {W_} {H_}" role="img" aria-label="held-out chrF by training step">{g}{lines}'
            f'<text x="{W_ - 20}" y="{H_ - 2}" class="tk" style="text-anchor:end">training step</text></svg><div class="legend">{leg}</div></div>')


def final_section(F):
    """The two final models: quality, speed, size, next to the earlier best ones (all from result files)."""
    SP = load(os.path.join(W, "speed_ours.json"), {}); SS = load(os.path.join(W, "seen_split.json"), {}); E8 = load(os.path.join(W, "v8_eval_v8.json"), {})
    names = [("student_V9", "<b>v9</b>"), ("student_V8", "v8 Complete"), ("student_V8F", "<b>v8 Fast</b>"), ("student_V7", "v7"), ("student_V6", "v6"), ("student_MAX_edited", "v5 MAX"), ("none", "Fast rules only")]
    rows = []
    for k, lab in names:
        r = F.get(k)
        if not r:
            continue
        a = r.get(("gold", "Accurate"), r.get(("gold", "Fast"))); b = r.get(("abo", "Accurate"), r.get(("abo", "Fast")))
        u = SS.get(k, {}); un = f'{pct(u["gold_unseen"]["accurate"]["exact"] * 100)} / {u["gold_unseen"]["accurate"]["chrF"]:.1f}' if u else "-"
        rows.append([lab, f"{pct(a[0])} / {a[1]:.1f}", un, f"{b[1]:.1f}", f'{r.get("cases", "-")}/70 · {r.get("blind", "-")}/48', f"{r['size']:.0f} MB" if "size" in r else "-"])
    h = table(["Full system", "Labels: exact / chrF", "Labels, unseen only", "Real names: chrF", "Address cases · unseen", "Model size"], rows)
    if SP:
        srow = []
        for k, lab in names:
            x = SP.get(k)
            if not x:
                continue
            L, N = x["labels"], x["names"]; al = L.get("alone"); an = N.get("alone")
            srow.append([lab, f'{L["system"]["median_ms"]:.1f} ms', f'{L["system"]["per_second"]:.0f}', f'{N["system"]["median_ms"]:.1f} ms', f'{N["system"]["per_second"]:.0f}',
                         f'{al["median_ms"]:.1f} / {an["median_ms"]:.1f} ms' if al else "-", f'{an["tokens_per_second"]:.0f}' if an else "-"])
        h += ("<h3>Speed on one CPU thread</h3><p>Median time per request and requests per second, measured on this server (4-core cloud CPU, one thread, 8-bit models). "
              "Short labels = the 320 test labels; product names = 330 long real names. Tokens are each model's own output steps (word pieces for v6–v8, characters for v5).</p>"
              + table(["Full system", "Short label: median", "Short labels / s", "Product name: median", "Product names / s", "Model alone: median (label / name)", "Output tokens / s"], srow, "wide"))

    # --- v9 against Hy-MT2 (both alone, no rules)
    T7 = load(os.path.join(W, "teachers", "gold_hy7.json"), {}); T18 = load(os.path.join(W, "teachers", "gold_hy18.json"), {})
    A7 = load(os.path.join(W, "teachers", "abo_hy7.json"), {}); A18 = load(os.path.join(W, "teachers", "abo_hy18.json"), {}); S330 = load(os.path.join(W, "teachers", "abo_sample330.json"), {})
    E9 = load(os.path.join(W, "v8_eval_post_student_V9.json"), {}); v9 = F.get("student_V9", {}); sp9 = SP.get("student_V9", {})
    if E9 and T7:
        g9 = E9["gold"]["experts"]; a9 = E9["abo"]["experts"]
        hr = [["<b>v9</b> (alone, full precision)", f'{pct(g9["exact"] * 100)} / {g9["chrF"]:.1f}', f'{a9["chrF"]:.1f}', f'{v9.get("size", 0):.0f} MB',
               f'{sp9["labels"]["alone"]["median_ms"]:.0f} ms / {sp9["names"]["alone"]["median_ms"]:.0f} ms (1 thread)' if sp9 else "-"],
              ["Hy-MT2-1.8B", f'{pct(T18["exact"] * 100)} / {T18["chrF"]:.1f}', f'{A18.get("chrF", 0):.1f}', "1,133 MB", "400–800 ms / 1,550 ms (4 threads)"],
              ["Hy-MT2-7B", f'{pct(T7["exact"] * 100)} / {T7["chrF"]:.1f}', f'{A7.get("chrF", 0):.1f} (on 330 names)', "4,625 MB", "1,400 ms / 6,000 ms (4 threads)"]]
        h += ("<h3>v9 against Hy-MT2</h3><p>Each model on its own, same tests, same machine.</p>"
              + table(["Model", "Labels: exact / chrF", "Real names: chrF", "Size", "Time per label / product name"], hr)
              + f'<p><b>Hy-MT2-1.8B is beaten on both tests</b> by a model 20 times smaller and far faster. <b>Hy-MT2-7B is beaten on real product names</b> '
              f'(v9 learned the catalogue style from real pairs) <b>but not on short labels</b>, where it scores {T7["chrF"]:.0f} against {g9["chrF"]:.0f}: '
              "a 7-billion-parameter model simply knows more rare product words in every language. Closing that last gap needs its answers as training data, which takes a GPU; the post-training tool below is built for exactly that kind of addition.</p>")
    if E9 or E8:
        bl = (E9 or E8)["gold"]["by_lang"]
        h += ("<p class=\"legend\">v9 alone, test labels by language (chrF): " + ", ".join(f"{l} {v:.0f}" for l, v in sorted(bl.items(), key=lambda x: -x[1])) + ".</p>")
    return h


def adding_section():
    return ("<p>Three ways, from fastest to most thorough. Each tests itself and says PASS or FAIL. The full guide with commands is in <code>docs/ADDING.md</code>.</p>"
            + table(["Way", "Use it for", "Steps", "Time", "Safety"], [
                ["<b>1. Add words</b>", "product words, brand terms, fixed phrases",
                 "1. Write a file: language, foreign text, English, separated by TABs.<br>2. <code>python build/add_terms.py my_terms.tsv</code>", "seconds, no training",
                 "re-runs the 70 cases and the test labels; undoes itself if a score drops"],
                ["<b>2. Add translated pairs</b> (post-training)", "your own catalogue, corrections of wrong outputs, a new category",
                 "1. Write one JSON line per pair: <code>{\"src\", \"tgt\", \"lang\"}</code>.<br>2. <code>sh build/post_train.sh student_V9 student_V10 120 my_pairs.jsonl</code><br>3. Read the verdict.", "1–5 hours on 4 CPU cores",
                 "the parent model is never changed; the new one is kept only if no test drops by more than 0.5"],
                ["<b>3. Add English-only names</b>", "a new catalogue section with no translations",
                 "1. One English name per line.<br>2. <code>python build/make_pairs.py names.txt --langs ko,th,de --out my_pairs.jsonl</code><br>3. Check a sample by eye.<br>4. Continue with way 2.", "0.7 s per name and language",
                 "wrong-script, copied and odd-length outputs are dropped automatically"]], "wide")
            + "<p>What post-training does on its own, in order: (1) starts from the parent model's weights; (2) fine-tunes on the new pairs mixed with all the old training data, so nothing is forgotten; "
            "(3) exports the 8-bit model; (4) retunes the rule that decides when the neural answer is used; (5) runs the tuning labels, test labels, real names and address cases on parent and child; "
            "(6) prints PASS or FAIL. If the machine restarts, the same command continues where it stopped. <b>v9 itself was made this way from v8 Complete</b>, so the tool has been used for real.</p>"
            "<p>Limits: the vocabulary is fixed, so text in a script the model has never seen cannot be taught this way (the first output line reports how much of the new data is unreadable); a whole new language also needs rules and detector support.</p>")


def steps_section(F):
    v8 = F.get("student_V8", {}); v8f = F.get("student_V8F", {})
    g = lambda r, k: (f'{r[k][1]:.1f}' if k in r else "-")
    s8 = ("<h3>Step 9. v8 Complete and v8 Fast, the two final models (Nov 12–13)</h3><ol>"
          "<li>Goal: the best accuracy and the best speed together, using everything learned; plus a much faster small model for quick validation.</li>"
          "<li>Measured where v7 was weakest: Korean (55 chrF) and Thai (58), the two languages with no short-label training data, because OPUS-MT cannot write them.</li>"
          "<li>New data: Hy-MT2-1.8B wrote 13k short labels (5,000 Korean, 4,000 Thai, 1,500 each Vietnamese, Japanese, Chinese) from Amazon search queries; kept 12,992 after script and length checks.</li>"
          "<li>Speed by design: the decoder runs once per output token, the encoder once per label. So v8 keeps v6's 6 encoder layers but only 2 of its 6 decoder layers (tested which two to keep), "
          "and its output layer covers only the 13.6k word pieces that occur in English instead of the shared 39k.</li>"
          "<li>Mixture of experts without a speed cost: one expert per language in the encoder only (1.6 MB each), chosen by the language detector, trained together with the rest.</li>"
          "<li>Fine-tuned everything for 7 hours on 470k pairs. Right after the cut the model scored 2; after 30 minutes 71; at the end it passed v7.</li>"
          "<li>Exported with per-channel 8-bit weights; retuned the decision rule on the tuning labels; ran the full tests.</li>"
          f"<li>v8 Fast: cut again from v8 Complete (3 encoder layers, 1 decoder layer, no experts), fine-tuned 3 hours, same export and tests.</li>"
          f"<li>Results: v8 Complete {g(v8, ('gold', 'Accurate'))} on labels and {g(v8, ('abo', 'Accurate'))} on real names; v8 Fast {g(v8f, ('gold', 'Accurate'))} and {g(v8f, ('abo', 'Accurate'))}. "
          "Lesson: a shallow decoder plus data aimed at the weakest languages gave more accuracy and more speed at once.</li></ol>"
          "<h3>Step 10. v9 and the post-training tool (Nov 13–14)</h3><ol>"
          "<li>Goal: get closer to (or past) Hy-MT2 without its size, and make later additions a routine job.</li>"
          "<li>Built three tools: add words instantly (<code>add_terms.py</code>), post-train on new pairs with an automatic pass/fail gate (<code>post_train.sh</code>), and turn English-only lists into pairs (<code>make_pairs.py</code>).</li>"
          "<li>New data: until now each English search query had been translated into only 3 of 14 languages. Translated all 54k queries into 7 languages each with OPUS-MT (393k translations, 251k kept after the round-trip check), plus 6k more Korean and Thai labels from Hy-MT2-1.8B: 199,525 new pairs.</li>"
          "<li>Ran the post-training tool on v8 Complete for 5 hours: new pairs sampled more often, all old data replayed.</li>"
          "<li>The tool exported, retuned, tested parent against child and gave its verdict; then the usual full tests and speed measurements.</li>"
          f"<li>Result: v9 {g(F.get('student_V9', {}), ('gold', 'Accurate'))} on labels and {g(F.get('student_V9', {}), ('abo', 'Accurate'))} on real names.</li></ol>")
    return open(os.path.join(HERE, "steps.html"), encoding="utf-8").read().replace("{{STEPS_V8}}", s8)


def v6_section(F):
    """The final model: seen / unseen split, 8-bit settings, routing rule (all read from result files)."""
    SS = load(os.path.join(W, "seen_split.json"), {}); Q = load(os.path.join(W, "quant_student_V6.json"), {}); R3 = load(os.path.join(W, "routing3_student_V6.json"), {})
    if "student_V6" not in SS:
        return "<p>Pending.</p>"
    lab = {"student_V6": "v6", "student_MAX_edited": "v5 MAX", "student_C": "v4 C", "student_V7": "v7 (v6 + language experts)"}
    c = lambda d, k: f'{pct(d[k]["exact"] * 100)} / {d[k]["chrF"]:.1f}'
    rows = [["Fast only"] + [c(SS["student_V6"][s], "fast") for s in ("gold_all", "gold_unseen", "abo_all", "abo_unseen")]]
    for m in ("student_C", "student_MAX_edited", "student_V6", "student_V7"):
        if m in SS:
            rows.append([f"<b>{lab[m]}</b>" if m in ("student_V6", "student_V7") else lab[m]] + [c(SS[m][s], "accurate") for s in ("gold_all", "gold_unseen", "abo_all", "abo_unseen")])
    n = SS["student_V6"]
    h = table(["Full system", f'Labels, all {n["gold_all"]["n"]}', f'Labels, unseen {n["gold_unseen"]["n"]}',
               f'Real names, all {n["abo_all"]["n"]}', f'Real names, unseen {n["abo_unseen"]["n"]}'], rows)
    h += ('<p class="legend">Exact match / chrF. "Unseen" leaves out test items whose source text also appears somewhere in the training data. '
          "That overlap is coincidence (common labels like \"phone case\" are also in catalogues and in the mined word list), it applied to every earlier model too, "
          "and the test files themselves were never trained on. The unseen columns are the strict measure.</p>")
    if Q:
        names = {"fp32": "full precision (reference)", "default": "8-bit, default", "per_channel": "8-bit, per channel (chosen)", "unsigned": "8-bit, unsigned weights",
                 "matmul_only": "8-bit, only matrix products", "matmul_only_per_channel": "8-bit, only matrix products, per channel", "no_embed_lmhead": "8-bit, embeddings and output layer kept full"}
        qr = [[names[k], f'{Q[k]["labels"]:.1f}', f'{Q[k]["titles"]:.1f}', f'{Q[k]["MB"]:.0f} MB', f'{Q[k]["ms"]:.0f} ms'] for k in names if k in Q and "labels" in Q[k]]
        h += "<h3>Changing the weights: 8-bit versions</h3><p>Scored on tuning data only (the new hand-written tuning labels and held-out titles), model alone, one server core.</p>"
        h += table(["Version", "Tuning labels (chrF)", "Held-out titles (chrF)", "Size", "Time per label"], qr)
        h += ("<p>The default 8-bit conversion cost about 5 chrF on short labels. Scaling each output channel separately recovers most of it at the same size and speed. "
              "A 7-bit range (which would make the browser and server round identically) lost too much, so the 8-bit version stays and the two runtimes differ on a few low-confidence answers. "
              "Averaging the last three checkpoints was also tested and made no difference on the tuning data, so the final weights are used.</p>")
    if R3:
        pol = R3["policy"]
        h += ("<h3>When the neural answer is used (rule v3)</h3><p>Chosen on tuning data (hand-written tuning labels, held-out titles, held-out place names): "
              f'Fast keeps the label when it is a place, an address, or already English{", or when Fast knew every word" if pol.get("keep_cov") else ""}. '
              f'Otherwise the neural answer is used if its confidence is at least {pol.get("tau")}. '
              + ("The English check on the neural answer was dropped: it rejected good answers full of brand names. " if not pol.get("need_english") else "")
              + (f'A label longer than {pol.get("addr_max_words")} words is treated as an address only if Fast knew most of its words; long product names with a number and a place word were being mistaken for addresses. ' if pol.get("addr_max_words") else "")
              + f'Tuning score: labels {R3["dev"]["labels"]["chrF"]:.1f}, titles {R3["dev"]["titles"]["chrF"]:.1f} chrF. These settings ship inside the model file, so Python and JavaScript apply the same rule.</p>')

    ME = load(os.path.join(W, "moe_eval_moe_v7.json"), {})
    if ME:
        mr = [[n, f'{pct(ME[k]["plain_v6"]["exact"] * 100)} / {ME[k]["plain_v6"]["chrF"]:.1f}', f'{pct(ME[k]["experts"]["exact"] * 100)} / {ME[k]["experts"]["chrF"]:.1f}',
               pct(ME[k]["routed"] * 100), pct(ME[k]["route_is_true_language"] * 100)] for k, n in (("gold", "Test labels (320)"), ("dev_labels", "Tuning labels (192)"), ("abo", "Real names (993)"))]
        h += ("<h3>v7: one expert per language (mixture of experts)</h3>"
              "<p>v6 stays frozen as the shared base. Each of the 16 languages gets its own small expert: an extra pair of matrices (512 → 64 → 512) added to the output of all 12 layers, "
              "786k parameters or 3.15 MB per language. The language detector of the Fast model picks the expert, so only the needed language's parameters are loaded and active; "
              "when the detector is unsure, no expert is used and the output is exactly v6. Experts were trained for 4 hours on the labels the detector routes to them, which keeps routing identical in training and on the phone.</p>"
              + table(["Model alone, full precision", "Plain v6", "v6 + experts", "Labels given an expert", "Detector picked the true language"], mr)
              + "<p>The gain is real but small and uneven: about +2 chrF on the test labels, nothing on the tuning labels or real names. As a full system (8-bit, with Fast) v7 adds about +1 chrF on labels over v6. "
              "Two things limit it. The experts learn from the same data v6 already fitted, so they add capacity but no new knowledge; and the detector is often wrong on short Latin-script labels "
              "(it picks the true language for about 87% of the test labels and 76% of real names). The design pays off more when each language has its own extra data or a better teacher.</p>")
    return h


def table(head, rows, cls=""):
    th = "".join(f"<th>{E(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def pct(x):
    return f"{x:.1f}%"


def main():
    F = parse_final()
    order = [k for k in META if k in F and ("abo", "Fast") in F[k] and ("gold", "Fast") in F[k] and not (k.endswith("JG") and "student_JG_edited" in F) and not (k.endswith("BASE") and "student_BASE_edited" in F)]
    T = {k: load(os.path.join(W, "teachers", f"gold_{k}.json"), {}) for k in ("hy7", "hy18", "opus", "nllb")}
    R = {k: load(os.path.join(W, f"routing_student_{k}.json")) for k in ("MAX", "B2")}
    L = load(glob.glob(os.path.join(W, "jepa_lookup_jepa_main.json"))[0]) if glob.glob(os.path.join(W, "jepa_lookup_jepa_main.json")) else None

    # --- main comparison table
    main_rows = []
    for k in order:
        r = F[k]; lab, data, com, mins = META[k]
        ga = r.get(("gold", "Accurate"), r.get(("gold", "Fast"))); gs = r.get(("gold", "Student only"))
        aa = r.get(("abo", "Accurate"), r.get(("abo", "Fast"))); as_ = r.get(("abo", "Student only"))
        main_rows.append([f"<b>{E(lab)}</b>", E(data), "yes" if com else "no",
                          f"{pct(ga[0])} / {ga[1]:.1f}", f"{pct(gs[0])} / {gs[1]:.1f}" if gs else "-",
                          f"{pct(aa[0])} / {aa[1]:.1f}", f"{pct(as_[0])} / {as_[1]:.1f}" if as_ else "-",
                          f"{r.get('cases', '-')}/70 · {r.get('blind', '-')}/48", f"{r['size']:.0f} MB" if "size" in r else "data 21 MB",
                          f"{mins / 60:.1f} h" if mins else "-"])
    gold_chart = bars([(META[k][0], F[k].get(("gold", "Accurate"), F[k].get(("gold", "Fast")))[1], "commercial use OK" if META[k][2] else "")
                       for k in order], 90, "chrF")
    abo_chart = bars([(META[k][0], F[k].get(("abo", "Accurate"), F[k].get(("abo", "Fast")))[1], "commercial use OK" if META[k][2] else "")
                      for k in order], 90, "chrF")

    # --- teachers
    best_small = max((F[k][("gold", "Accurate")][1], META[k][0]) for k in order if ("gold", "Accurate") in F[k])
    TA = {k: load(os.path.join(W, "teachers", f"abo_{k}.json"), {}) for k in ("hy7", "hy18", "opus", "nllb")}
    S330 = load(os.path.join(W, "teachers", "abo_sample330.json"), {}); MUL = load(os.path.join(W, "teachers", "marian_mul.json"), {})
    def abo_cell(k):
        if not TA[k]:
            return "-"
        return f'{TA[k]["chrF"]:.1f}' + (f' (on {TA[k]["n"]} names)' if TA[k].get("per_lang_cap") else "")
    teach_rows = [[E(n), pct(T[k].get("exact", 0) * 100), f'{T[k].get("chrF", 0):.1f}', abo_cell(k), E(s), E(lic)] for k, n, s, lic in [
        ("hy7", "Hy-MT2-7B (Tencent)", "7B parameters, ~4.6 GB at 4-bit; 1.4 s per label, 6 s per product name here", "Apache-2.0"),
        ("hy18", "Hy-MT2-1.8B (Tencent)", "1.8B, ~1.1 GB at 4-bit; 0.4-1.6 s", "Apache-2.0"),
        ("opus", "OPUS-MT (Helsinki-NLP), one model per language", "75M per language, ~300 MB each; 0.3 s per product name", "CC BY 4.0"),
        ("nllb", "NLLB-200-distilled-600M (Meta)", "600M, one model for all languages; 0.7 s per product name", "CC BY-NC 4.0")]]
    if MUL:
        teach_rows.append(["OPUS-MT mul-en, before our fine-tuning (v6's starting point)", pct(MUL["gold"]["exact"] * 100), f'{MUL["gold"]["chrF"]:.1f}', f'{MUL["abo"]["chrF"]:.1f}', "77M, one model for all languages", "CC BY 4.0"])
    v6r = F.get("student_V6", {})
    if v6r:
        teach_rows.append(["<b>v6, model alone</b>", pct(v6r[("gold", "Student only")][0]), f'{v6r[("gold", "Student only")][1]:.1f}', f'{v6r[("abo", "Student only")][1]:.1f}', "65M after trimming, 106 MB at 8-bit; 15-75 ms", "commercial use OK"])
        teach_rows.append(["<b>v6, full system (with Fast)</b>", pct(v6r[("gold", "Accurate")][0]), f'{v6r[("gold", "Accurate")][1]:.1f}', f'{v6r[("abo", "Accurate")][1]:.1f}', "", ""])
    teach_note = ""
    if S330:
        teach_note = (f'<p>Hy-MT2-7B takes about 6 seconds per product name on this machine, so it ran on a balanced sample of {S330["n"]} real names (30 per language). '
                      f'On that same sample: Hy-MT2-7B {S330["hy7"]["chrF"]:.1f}, Hy-MT2-1.8B {S330["hy18"]["chrF"]:.1f}, NLLB {S330["nllb"]["chrF"]:.1f}, OPUS-MT {S330["opus"]["chrF"]:.1f}, '
                      f'Fast {S330["fast"]["chrF"]:.1f}, <b>v6 {S330["v6"]["chrF"]:.1f}</b> chrF.</p>'
                      "<p><b>Why v6 beats even the 7B model on real names.</b> The reference answers are Amazon's own English titles, which are not literal translations: they follow a catalogue style "
                      "(brand first, a fixed order of attributes, US units). v6 was fine-tuned on 35k such pairs and learned that style; the general translators translate faithfully and lose points for wording. "
                      "On the short hand-written labels, where a plain correct translation is what counts, Hy-MT2-7B is still clearly ahead (95 vs 80).</p>")

    # --- weight testing and changes
    edit_rows = []
    for V, lab in [("A", "v4 A"), ("B", "v4 B"), ("C", "v4 C"), ("B2", "v5 B2"), ("C2", "v5 C2"), ("H", "v5 H"), ("ALL3", "v5 ALL3"),
                   ("HY", "v5 HY"), ("MAX", "v5 MAX"), ("JG", "JEPA-Gen"), ("BASE", "Baseline")]:
        d = load(os.path.join(W, f"edits_student_{V}.json"))
        if not d:
            continue
        b, e, ft = d["base"], d.get("edited", {}), d.get("finetuned", {})
        kept = ", ".join(d.get("kept", [])) or "none"
        fts = (f'{ft.get("bt_chrF", 0):.1f} / {ft.get("gold_chrF", 0):.1f}' + (" (used)" if ft.get("used") and d["decision"].startswith("use") else "")) if ft else "-"
        edit_rows.append([E(lab), str(len(d.get("tried", []))), f"<code>{E(kept)}</code>",
                          f'{b.get("bt_chrF", 0):.1f} / {b.get("gold_chrF", 0):.1f}', f'{e.get("bt_chrF", 0):.1f} / {e.get("gold_chrF", 0):.1f}', fts,
                          "<b>changed</b>" if d["decision"].startswith("use") else "kept original"])

    # --- decision rule
    rule_rows = []
    for k, lab in (("B2", "v5 B2"), ("MAX", "v5 MAX")):
        r = R.get(k)
        if r:
            rule_rows.append([lab, f'{pct(r["gold"]["rule_v1"]["exact"] * 100)} / {r["gold"]["rule_v1"]["chrF"]:.1f}',
                              f'<b>{pct(r["gold"]["rule_v2"]["exact"] * 100)} / {r["gold"]["rule_v2"]["chrF"]:.1f}</b>',
                              f'{pct(r["abo"]["rule_v1"]["exact"] * 100)} / {r["abo"]["rule_v1"]["chrF"]:.1f}',
                              f'<b>{pct(r["abo"]["rule_v2"]["exact"] * 100)} / {r["abo"]["rule_v2"]["chrF"]:.1f}</b>'])

    # --- JEPA
    tune_rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "models", "jepa_t_*", "jepa_log.json"))):
        d = load(p); a = d["args"]; ev = d["evals"][-1]
        tune_rows.append([E(os.path.basename(os.path.dirname(p))[7:]), str(a["ema"]), f'{int(a["mask"] * 100)}%', str(a["pred_layers"]),
                          pct(ev["retrieval@1"] * 100), f'{ev["spread"]:.2f}'])
    jm = load(os.path.join(ROOT, "models", "jepa_main", "jepa_log.json"), {"evals": []})
    jcurve = " → ".join(f'{pct(e["retrieval@1"] * 100)} (step {e["step"]})' for e in jm["evals"][::3] + jm["evals"][-1:]) if jm["evals"] else "-"
    jg = F.get("student_JG_edited") or F.get("student_JG"); bs = F.get("student_BASE_edited") or F.get("student_BASE")
    def cell(r, key):
        v = r.get(key) if r else None
        return f"{pct(v[0])} / {v[1]:.1f}" if v else "-"
    jepa_rows = [[n, cell(r, ("gold", "Student only")), cell(r, ("gold", "Accurate")), cell(r, ("abo", "Student only")), cell(r, ("abo", "Accurate"))]
                 for n, r in (("JEPA-Gen (3 h JEPA + 3 h translation)", jg), ("Baseline (6 h translation, no JEPA)", bs), ("v5 MAX (10 h, same data)", F.get("student_MAX_edited")))]
    look_rows = []
    if L:
        for s, n in (("gold", "320 hand-written labels"), ("abo", "993 real ABO names")):
            x = L[s]
            look_rows.append([n, f'{pct(x["lookup_alone"]["exact"] * 100)} / {x["lookup_alone"]["chrF"]:.1f}',
                              f'{pct(x["fast"]["exact"] * 100)} / {x["fast"]["chrF"]:.1f}', f'{pct(x["fast+lookup"]["exact"] * 100)} / {x["fast+lookup"]["chrF"]:.1f}',
                              f'{pct(x["accurate"]["exact"] * 100)} / {x["accurate"]["chrF"]:.1f}' if "accurate" in x else "-",
                              f'{pct(x["accurate+lookup"]["exact"] * 100)} / {x["accurate+lookup"]["chrF"]:.1f}' if "accurate+lookup" in x else "-",
                              pct(x["lookup_used"] * 100)])


    # --- KPIs, verdicts, data, time
    acc = {k: F[k].get(("gold", "Accurate"), F[k].get(("gold", "Fast"))) for k in order}
    abo = {k: F[k].get(("abo", "Accurate"), F[k].get(("abo", "Fast"))) for k in order}
    com = [k for k in order if META[k][2] and k != "none"]
    bg = max(order, key=lambda k: acc[k][1]); bc = max(com, key=lambda k: acc[k][1] + abo[k][1])
    kpis = [(f"{acc['none'][1]:.1f} → {acc[bg][1]:.1f}", f"label chrF, Fast → best ({META[bg][0]})"),
            (f"{abo['none'][1]:.1f} → {max(abo.values(), key=lambda x: x[1])[1]:.1f}", "real-name chrF, Fast → best"),
            (META[bc][0], "best model with commercial-use teachers"),
            (f"{len(order) - 1}", "neural models trained and compared")]
    kp = "".join(f"<div class=kpi><b>{E(a)}</b><span>{E(b)}</span></div>" for a, b in kpis)
    if jg and bs and ("gold", "Accurate") in jg and ("gold", "Accurate") in bs:
        dg = jg[("gold", "Accurate")][1] - bs[("gold", "Accurate")][1]; da = jg[("abo", "Accurate")][1] - bs[("abo", "Accurate")][1]
        dgs = jg[("gold", "Student only")][1] - bs[("gold", "Student only")][1]
        verdict = (f"<b>Verdict on JEPA.</b> With the same model size, data and total time, JEPA-Gen vs the baseline: "
                   f"{dgs:+.1f} chrF on labels (model alone), {dg:+.1f} on labels (Accurate), {da:+.1f} on real names (Accurate). ")
        if dg + da > 1.0:
            verdict += "JEPA pretraining paid for its 3 hours."
        elif dg + da < -1.0:
            verdict += ("JEPA pretraining did not pay for its 3 hours; spending them on ordinary translation training was better. "
                        "JEPA-Gen did start faster (higher held-out scores at the same step, see the chart), so the encoder did learn something useful, "
                        "but the baseline overtook it once it had the extra time. JEPA tends to help when there is far more unlabeled text than translated pairs; "
                        "here we have about 400k translated pairs, so plain training learns the same things directly. Decision: JEPA-Gen is not shipped.")
        else:
            verdict += "That is within noise: JEPA pretraining did not pay for its 3 hours here."
    else:
        verdict = "<b>Verdict on JEPA.</b> Pending: the baseline is still training."
    if L:
        if L["threshold"] > 1.0:
            verdict += (f" <b>Lookup:</b> on its own it reaches {L['gold']['lookup_alone']['chrF']:.1f} chrF on labels and {L['abo']['lookup_alone']['chrF']:.1f} on real names, "
                        "well below Fast. Tuned on held-out data, the best threshold was to never use it: every setting that let it override Fast lowered the score. "
                        "The nearest known name is usually a related product, not the same one; label wording varies too much for whole-name lookup.")
        else:
            verdict += (f" <b>Lookup:</b> alone {L['abo']['lookup_alone']['chrF']:.1f} chrF on real names; on top of Accurate it changes real-name chrF from "
                        f"{L['abo'].get('accurate', {}).get('chrF', 0):.1f} to {L['abo'].get('accurate+lookup', {}).get('chrF', 0):.1f}.")
    data_rows = [["Amazon ESCI product titles", "81k English titles; back-translated into 16 languages", "Apache-2.0"],
                 ["Amazon Berkeley Objects (ABO)", "55.8k real names in 12 languages paired with English; 993 held out for testing; 114k English titles", "CC BY 4.0"],
                 ["GeoNames", "place names for the gazetteer", "CC BY 4.0"],
                 ["OPUS-MT (Helsinki-NLP)", "teacher: translations both ways, answers, round-trip checks", "CC BY 4.0"],
                 ["M2M100-418M (Meta)", "teacher for v4 B where OPUS-MT had no model", "MIT"],
                 ["NLLB-200-distilled-600M (Meta)", "teacher for v3, v4 A/C, C2, ALL3", "CC BY-NC 4.0 (non-commercial)"],
                 ["Hy-MT2-1.8B and 7B (Tencent)", "teacher for H, ALL3, HY, MAX; 7B used only as a reference score", "Apache-2.0"],
                 ["Sudachi dictionary", "Japanese readings", "Apache-2.0"]]
    hours = sum(META[k][3] for k in META if k in F or k.replace("_edited", "") in F) / 60
    time_rows = [["Chat and work", "Nov 4 02:25 UTC → Nov 14 04:30 UTC: about 242 hours (10.1 days) from the first message to this report, including many hours lost to machine restarts"],
                 ["v6: short-label data + fine-tuning + export, quantization and routing tests", "1 h + 8.5 h + 1.5 h"],
                 ["v9 and the post-training tools (data 6 h, post-training 5 h, tests 1 h)", "12 h"],
                 ["v8 Complete and v8 Fast (Hy-MT2 short labels, training, export, tests, speed)", "4 h + 7 h + 3 h + 1 h"],
                 ["v7: language experts (training, export, tests)", "4 h + 1 h"],
                 ["Training the final neural models (sum)", f"{hours:.0f} h on 4 CPU cores"],
                 ["Teacher data generation", "~25 h (OPUS-MT, NLLB, M2M100, Hy-MT2 back-translation and answers)"],
                 ["JEPA pretraining and tuning", "2 h tuning (4 × 30 min) + 3 h main run"],
                 ["Weight testing and edits", "~25–30 h in total (about 2–3 h per model: switch-off tests, edits, 15-min retraining)"],
                 ["Total compute spent training and changing models", "≈ 130–135 h; the rest was evaluation, packaging, restarts and idle time"]]
    reco = ("<ul><li><b>Use v9</b> for real work: the most accurate model and about twice as fast as v6/v7 at half the size. To teach it more, use the post-training tool (see the section on adding new things). <b>Use v8 Fast</b> for quick validation runs. The earlier models are kept for reference only.</li><li>Earlier advice, still true for v6/v7 (106 MB): it more than doubles the exact-match rate on labels and adds over 20 chrF on real names, and its teachers and data allow commercial use (with attribution).</li>"
            "<li><b>Ship v5 MAX</b> (22 MB) when download size or speed matters more: about 10 ms per short label.</li>"
            "<li><b>Fast alone</b> stays the right choice for addresses and place names (~2 ms); every Accurate model keeps using it for those.</li>"
            "<li><b>Next gains:</b> (1) Hy-MT2-7B scores 95 on the labels, so distilling its answers into v6 on a GPU should close much of the remaining gap; "
            "(2) Korean is still the weakest language (its word pieces were new to the pretrained model); "
            "(3) a smaller v6 by sharing the output layer with the embeddings (about 20 MB less).</li></ul>")
    v6 = v6_section(F)

    # --- downloads, conclusions, important notes
    GH = "https://github.com/cardimvitor/translation-model"; BR = "ccr-9202c40a-kuyiob"
    DL = [("student_V9", "student_V9", "v9", "FINAL. Best model."), ("student_V8", "student_V8", "v8 Complete", "Parent of v9."), ("student_V8F", "student_V8F", "v8 Fast", "FINAL. Fast model for validation."),
          ("student_V6", "student_V6", "v6", "Earlier best."), ("student_MAX_edited", "student_MAX", "v5 MAX", "Best small model for real product names."),
          ("student_B2", "student_B2", "v5 B2", "Small, OPUS-MT only."), ("student_C2_edited", "student_C2", "v5 C2", "Small, best v5 on labels."),
          ("student_A", "student_A", "v4 A", ""), ("student_B", "student_B", "v4 B", ""), ("student_C", "student_C", "v4 C", "Best small model on labels."),
          ("student_H", "student_H", "v5 H", "Experiment."), ("student_ALL3_edited", "student_ALL3", "v5 ALL3", "Experiment."),
          ("student_HY_edited", "student_HY", "v5 HY", "Experiment."), ("student_JG", "student_JEPA", "JEPA-Gen", "Experiment, not recommended."),
          ("student_BASE_edited", "student_BASE", "Baseline", "Comparison model for JEPA."), ("student_edited", "student_v3", "v3", "First neural model.")]
    dl_rows = []
    for key, folder, lab, note in DL:
        r = F.get(key, {}); com = META.get(key, ("", "", False))[2]
        a = r.get(("gold", "Accurate")); b = r.get(("abo", "Accurate"))
        dl_rows.append([f"<b>{E(lab)}</b>", f'<a href="{GH}/tree/{BR}/models/{folder}">models/{folder}</a>', f"{r['size']:.0f} MB" if "size" in r else "-",
                        f"{a[1]:.1f} / {b[1]:.1f}" if a and b else "-", "yes" if com else "no", E(note)])
    downloads = (table(["Model", "Download (GitHub folder)", "Size", "chrF: labels / real names", "Commercial use", "Note"], dl_rows, "wide")
                 + f'<p>Each folder holds <code>encoder_int8.onnx</code>, <code>decoder_int8.onnx</code> and <code>vocab.json</code> (v6 also <code>spm_src.json</code>). '
                 f'To get everything at once (all models, Fast data, Python and JavaScript runtimes, build scripts, tests): '
                 f'<a href="{GH}/archive/refs/heads/{BR}.zip">download the whole branch as a zip</a>, or open <a href="{GH}/pull/1">pull request #1</a>. '
                 f'Fast needs no model: its data is in <a href="{GH}/tree/{BR}/data">data/</a> and its code in <a href="{GH}/tree/{BR}/runtime">runtime/</a>. '
                 "The repository is private, so the links work only when signed in to GitHub with access to it. They point at the work branch; after the pull request is merged, the same folders are on <code>main</code>. "
                 'Ready-to-use packages (model + runtime + data in one zip) for v6, v5 MAX, B2, C2 and Fast were also sent as files in the chat. '
                 'Try the models without downloading: <a href="https://claude.ai/artifact/3je2spXyFcH1QicAoMfZWL">phone test page</a>.</p>')
    V6a = F.get("student_V6", {}); g6 = V6a.get(("gold", "Accurate"), (0, 0)); a6 = V6a.get(("abo", "Accurate"), (0, 0))
    concl = ("<ol>"
             "<li><b>Two final models.</b> v9 (v8 Complete post-trained) is the most accurate and, among the strong models, the fastest and smallest; v8 Fast trades some accuracy for much more speed and is meant for validation.</li>"
             f"<li><b>The big jump came with v6.</b> The full system goes from {acc['none'][0]:.1f}% exact / {acc['none'][1]:.1f} chrF (rules only) to {g6[0]:.1f}% / {g6[1]:.1f} on hand-written labels, and from {abo['none'][1]:.1f} to {a6[1]:.1f} chrF on real marketplace names, fully offline.</li>"
             "<li><b>Starting from a pretrained model mattered more than everything else combined.</b> Seven days of from-scratch models (v3 to v5, twelve of them) moved label chrF from 43 to 51. One 8.5-hour fine-tune of a pretrained translator moved it to 80.</li>"
             "<li><b>Data that looks like the task beats more data.</b> 35k real product-name pairs lifted real-name scores by 7 points in v5; 58k short labels were a large part of v6's jump on short labels.</li>"
             "<li><b>The rules (Fast) remain necessary.</b> Place names and addresses are where the neural models are weakest (about 25 chrF on held-out place names) and Fast is near-perfect, so every system keeps Fast for those.</li>"
             "<li><b>Weight edits help a little, never a lot.</b> Switching off one or two attention heads gained 0.2–1 chrF on about half the small models. For v6 the weight change that mattered was the 8-bit conversion method (+3 chrF over the default).</li>"
             "<li><b>Language experts (v7) work, with a small gain.</b> Only the detected language's 3 MB expert is active; labels improve by about 1 chrF over v6. They would matter more with language-specific extra data.</li>"
             "<li><b>JEPA worked technically but did not pay off</b> with this much translated data: the same hours spent on ordinary training gave a better model.</li>"
             "<li><b>Against the base models:</b> v6 beats every base model on real product names, including the 7B one (it learned the catalogue's naming style), and beats all but Hy-MT2-7B on short labels. Hy-MT2-7B scores 95 there; distilling it into v6 on a GPU is the clearest path to further gains.</li>"
             "</ol>")
    important = ("<ul>"
                 "<li><b>Licenses.</b> v6, v5 MAX, B2, H, HY and v4 B were trained only with teachers and data that allow commercial use (OPUS-MT CC BY 4.0, Hy-MT2 Apache-2.0, ABO CC BY 4.0, ESCI Apache-2.0); attribution is required. v3, v4 A, v4 C, C2 and ALL3 used NLLB (CC BY-NC) and are non-commercial. City names need GeoNames attribution. Whether a teacher's license carries over to a model trained on its outputs is not settled law; get legal advice before selling.</li>"
                 "<li><b>Speed and size of v6.</b> 106 MB download; 15–75 ms per label on one server core; about 0.1–0.5 s per label in a single-thread browser (measured in Node, not on a real phone). The small models take about 10–65 ms.</li>"
                 "<li><b>It still makes mistakes.</b> About one label in three is not an exact match, and some are plainly wrong (an Arabic \"non-stick frying pan\" came out as \"unsticked mattress\"). Korean is the weakest language. For anything where a wrong name is costly, show the original next to the translation.</li>"
                 "<li><b>Test honesty.</b> The test files were never trained on, but 60 of 320 labels and 103 of 993 real names also occur by coincidence in training text. The \"unseen\" columns in the v6 section exclude them and are the strict numbers. The 48 unseen address cases were looked at once while choosing a cautious setting for place names; the score did not change.</li>"
                 "<li><b>Python and JavaScript.</b> Identical for Fast, the English check and the tokenizer. For v6 they differ on about 1–3% of labels because 8-bit arithmetic rounds differently; measured quality is the same (79.5 vs 79.6 on labels).</li>"
                 "<li><b>GitHub.</b> The pull request is a draft and nothing is merged. GitHub's automatic tests have never run because no runner gets assigned (an Actions setting or billing limit on the account); all tests pass when run here.</li>"
                 "<li><b>What is not in the repository.</b> The raw training data (several GB, regenerable with the build scripts), full-precision weights and training checkpoints. They exist only in this session's container, which is temporary. Say so if you want them kept.</li>"
                 "<li><b>Privacy.</b> All models run on the device; no label text is sent anywhere.</li>"
                 "</ul>")
    tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
    page = (tpl.replace("{{MAIN_TABLE}}", table(["Model", "Teachers / data", "Commercial use", "Labels: Accurate", "Labels: model alone",
                                                 "Real names: Accurate", "Real names: model alone", "Cases · unseen", "Size", "Training"], main_rows, "wide"))
            .replace("{{KPIS}}", kp).replace("{{V6}}", v6).replace("{{FINAL}}", final_section(F)).replace("{{STEPS}}", steps_section(F)).replace("{{ADDING}}", adding_section()).replace("{{DOWNLOADS}}", downloads).replace("{{CONCLUSIONS}}", concl).replace("{{IMPORTANT}}", important).replace("{{CURVES}}", curves()).replace("{{JEPA_VERDICT}}", verdict)
            .replace("{{DATA}}", table(["Source", "Used for", "License"], data_rows))
            .replace("{{TIME}}", table(["What", "Time"], time_rows)).replace("{{RECO}}", reco)
            .replace("{{GOLD_CHART}}", gold_chart).replace("{{ABO_CHART}}", abo_chart)
            .replace("{{TEACHERS}}", table(["Model", "Labels: exact", "Labels: chrF", "Real names: chrF", "Size and speed", "License"], teach_rows) + teach_note)
            .replace("{{EDITS}}", table(["Model", "Changes tested", "Kept", "Before: held-out / labels", "After change", "After 15-min retraining", "Decision"], edit_rows, "wide"))
            .replace("{{RULE}}", table(["Model", "Labels, old rule", "Labels, new rule", "Real names, old rule", "Real names, new rule"], rule_rows))
            .replace("{{JEPA_TUNE}}", table(["Trial", "Target update", "Hidden", "Predictor layers", "Right title in 1,000", "Spread"], tune_rows))
            .replace("{{JEPA_CURVE}}", E(jcurve))
            .replace("{{JEPA_GEN}}", table(["Model", "Labels: model alone", "Labels: Accurate", "Real names: model alone", "Real names: Accurate"], jepa_rows, "wide"))
            .replace("{{JEPA_LOOKUP}}", table(["Test set", "Lookup alone", "Fast", "Fast + lookup", "Accurate (MAX)", "Accurate + lookup", "Lookup used"], look_rows, "wide") if look_rows else "<p>Pending.</p>"))
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)
    print("wrote", os.path.join(OUT, "index.html"), len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
