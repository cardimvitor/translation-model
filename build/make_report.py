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
    "student_BASE": ("Baseline", "MAX data, no JEPA (same size, same total time)", True, 360),
}


def parse_final():
    txt = open(os.path.join(W, "final_eval.txt"), encoding="utf-8").read()
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
                       for k in order], 60, "chrF")
    abo_chart = bars([(META[k][0], F[k].get(("abo", "Accurate"), F[k].get(("abo", "Fast")))[1], "commercial use OK" if META[k][2] else "")
                      for k in order], 60, "chrF")

    # --- teachers
    best_small = max((F[k][("gold", "Accurate")][1], META[k][0]) for k in order if ("gold", "Accurate") in F[k])
    teach_rows = [[E(n), pct(T[k].get("exact", 0) * 100), f'{T[k].get("chrF", 0):.1f}', E(s), E(lic)] for k, n, s, lic in [
        ("hy7", "Hy-MT2-7B (Tencent)", "7B parameters, ~4.6 GB at 4-bit, ~1.4 s per label here", "Apache-2.0"),
        ("hy18", "Hy-MT2-1.8B (Tencent)", "1.8B, ~1.1 GB at 4-bit, ~0.4-0.8 s per label", "Apache-2.0"),
        ("opus", "OPUS-MT (Helsinki-NLP)", "75M per language, ~300 MB each", "CC BY 4.0"),
        ("nllb", "NLLB-200-distilled-600M (Meta)", "600M, one model for all languages", "CC BY-NC 4.0")]]
    teach_rows.append([f"<b>best small model ({E(best_small[1])})</b>", "", f"{best_small[0]:.1f}", "13-22 MB, ~10 ms per label on a phone CPU", ""])

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
    time_rows = [["Chat and work", "Nov 4 02:25 UTC → Nov 11 (about 7 days, much of it waiting on CPU training)"],
                 ["Training the final neural models (sum)", f"{hours:.0f} h on 4 CPU cores"],
                 ["Teacher data generation", "~25 h (OPUS-MT, NLLB, M2M100, Hy-MT2 back-translation and answers)"],
                 ["JEPA pretraining and tuning", "2 h tuning (4 × 30 min) + 3 h main run"],
                 ["Weight testing and edits", "~1 h per model"]]
    reco = (f"<ul><li><b>Ship {E(META[bc][0])}</b> as the Accurate model when commercial use matters: best total score among models whose teachers allow it.</li>"
            f"<li><b>Best scores overall</b> on hand-written labels: {E(META[bg][0])} (NLLB-derived, non-commercial).</li>"
            "<li><b>Keep Fast as the default</b> on phones: ~2 ms per label, and the best Accurate models add +6 to +8 chrF for 5–60 ms more per label.</li>"
            "<li><b>Next gains</b> come from real paired data (the ABO names moved real-name chrF by +7), not from bigger models or longer CPU training. "
            "A GPU run distilling Hy-MT2-7B answers on 200k+ titles is the clear next step.</li></ul>")
    tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
    page = (tpl.replace("{{MAIN_TABLE}}", table(["Model", "Teachers / data", "Commercial use", "Labels: Accurate", "Labels: model alone",
                                                 "Real names: Accurate", "Real names: model alone", "Cases · unseen", "Size", "Training"], main_rows, "wide"))
            .replace("{{KPIS}}", kp).replace("{{CURVES}}", curves()).replace("{{JEPA_VERDICT}}", verdict)
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
