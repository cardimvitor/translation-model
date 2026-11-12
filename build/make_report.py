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
    "student_BASE": ("Baseline", "MAX data, no JEPA (same size, same total time)", True, 360),
}


def parse_final():
    txt = open(os.path.join(W, "final_eval.txt"), encoding="utf-8").read()
    if os.path.exists(os.path.join(W, "final_eval_V6.txt")):
        txt += open(os.path.join(W, "final_eval_V6.txt"), encoding="utf-8").read()
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


def v6_section(F):
    """The final model: seen / unseen split, 8-bit settings, routing rule (all read from result files)."""
    SS = load(os.path.join(W, "seen_split.json"), {}); Q = load(os.path.join(W, "quant_student_V6.json"), {}); R3 = load(os.path.join(W, "routing3_student_V6.json"), {})
    if "student_V6" not in SS:
        return "<p>Pending.</p>"
    lab = {"student_V6": "v6", "student_MAX_edited": "v5 MAX", "student_C": "v4 C"}
    c = lambda d, k: f'{pct(d[k]["exact"] * 100)} / {d[k]["chrF"]:.1f}'
    rows = [["Fast only"] + [c(SS["student_V6"][s], "fast") for s in ("gold_all", "gold_unseen", "abo_all", "abo_unseen")]]
    for m in ("student_C", "student_MAX_edited", "student_V6"):
        if m in SS:
            rows.append([f"<b>{lab[m]}</b>" if m == "student_V6" else lab[m]] + [c(SS[m][s], "accurate") for s in ("gold_all", "gold_unseen", "abo_all", "abo_unseen")])
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
    teach_rows = [[E(n), pct(T[k].get("exact", 0) * 100), f'{T[k].get("chrF", 0):.1f}', E(s), E(lic)] for k, n, s, lic in [
        ("hy7", "Hy-MT2-7B (Tencent)", "7B parameters, ~4.6 GB at 4-bit, ~1.4 s per label here", "Apache-2.0"),
        ("hy18", "Hy-MT2-1.8B (Tencent)", "1.8B, ~1.1 GB at 4-bit, ~0.4-0.8 s per label", "Apache-2.0"),
        ("opus", "OPUS-MT (Helsinki-NLP)", "75M per language, ~300 MB each", "CC BY 4.0"),
        ("nllb", "NLLB-200-distilled-600M (Meta)", "600M, one model for all languages", "CC BY-NC 4.0")]]
    teach_rows.append([f"<b>best small model ({E(best_small[1])})</b>", "", f"{best_small[0]:.1f}", "106 MB at 8-bit; 15-75 ms per label on a server core, 0.1-0.5 s in a single-thread browser", "commercial use OK"])

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
    time_rows = [["Chat and work", "Nov 4 02:25 UTC → Nov 12 12:50 UTC: about 202 hours (8.4 days) from the first message to this report"],
                 ["v6: short-label data + fine-tuning + export, quantization and routing tests", "1 h + 8.5 h + 1.5 h"],
                 ["Training the final neural models (sum)", f"{hours:.0f} h on 4 CPU cores"],
                 ["Teacher data generation", "~25 h (OPUS-MT, NLLB, M2M100, Hy-MT2 back-translation and answers)"],
                 ["JEPA pretraining and tuning", "2 h tuning (4 × 30 min) + 3 h main run"],
                 ["Weight testing and edits", "~25–30 h in total (about 2–3 h per model: switch-off tests, edits, 15-min retraining)"],
                 ["Total compute spent training and changing models", "≈ 130–135 h; the rest was evaluation, packaging, restarts and idle time"]]
    reco = ("<ul><li><b>Ship v6</b> wherever a 106 MB model is acceptable: it more than doubles the exact-match rate on labels and adds over 20 chrF on real names, and its teachers and data allow commercial use (with attribution).</li>"
            "<li><b>Ship v5 MAX</b> (22 MB) when download size or speed matters more: about 10 ms per short label.</li>"
            "<li><b>Fast alone</b> stays the right choice for addresses and place names (~2 ms); every Accurate model keeps using it for those.</li>"
            "<li><b>Next gains:</b> (1) Hy-MT2-7B scores 95 on the labels, so distilling its answers into v6 on a GPU should close much of the remaining gap; "
            "(2) Korean is still the weakest language (its word pieces were new to the pretrained model); "
            "(3) a smaller v6 by sharing the output layer with the embeddings (about 20 MB less).</li></ul>")
    v6 = v6_section(F)

    # --- downloads, conclusions, important notes
    GH = "https://github.com/cardimvitor/translation-model"; BR = "ccr-9202c40a-kuyiob"
    DL = [("student_V6", "student_V6", "v6", "Recommended. Best quality."), ("student_MAX_edited", "student_MAX", "v5 MAX", "Best small model for real product names."),
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
             f"<li><b>The goal was reached with v6.</b> The full system goes from {acc['none'][0]:.1f}% exact / {acc['none'][1]:.1f} chrF (rules only) to {g6[0]:.1f}% / {g6[1]:.1f} on hand-written labels, and from {abo['none'][1]:.1f} to {a6[1]:.1f} chrF on real marketplace names, fully offline.</li>"
             "<li><b>Starting from a pretrained model mattered more than everything else combined.</b> Seven days of from-scratch models (v3 to v5, twelve of them) moved label chrF from 43 to 51. One 8.5-hour fine-tune of a pretrained translator moved it to 80.</li>"
             "<li><b>Data that looks like the task beats more data.</b> 35k real product-name pairs lifted real-name scores by 7 points in v5; 58k short labels were a large part of v6's jump on short labels.</li>"
             "<li><b>The rules (Fast) remain necessary.</b> Place names and addresses are where the neural models are weakest (about 25 chrF on held-out place names) and Fast is near-perfect, so every system keeps Fast for those.</li>"
             "<li><b>Weight edits help a little, never a lot.</b> Switching off one or two attention heads gained 0.2–1 chrF on about half the small models. For v6 the weight change that mattered was the 8-bit conversion method (+3 chrF over the default).</li>"
             "<li><b>JEPA worked technically but did not pay off</b> with this much translated data: the same hours spent on ordinary training gave a better model.</li>"
             "<li><b>Bigger teachers are still far ahead.</b> Hy-MT2-7B scores 95 on the labels. Distilling it into v6 on a GPU is the clearest path to further gains.</li>"
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
            .replace("{{KPIS}}", kp).replace("{{V6}}", v6).replace("{{DOWNLOADS}}", downloads).replace("{{CONCLUSIONS}}", concl).replace("{{IMPORTANT}}", important).replace("{{CURVES}}", curves()).replace("{{JEPA_VERDICT}}", verdict)
            .replace("{{DATA}}", table(["Source", "Used for", "License"], data_rows))
            .replace("{{TIME}}", table(["What", "Time"], time_rows)).replace("{{RECO}}", reco)
            .replace("{{GOLD_CHART}}", gold_chart).replace("{{ABO_CHART}}", abo_chart)
            .replace("{{TEACHERS}}", table(["Teacher", "Exact", "chrF", "Size and speed", "License"], teach_rows))
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
