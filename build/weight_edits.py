"""Test and change a trained student's weights; keep only edits that help on data not used to find them.

  python weight_edits.py <model_dir> [--dev pairs_dev.jsonl] [--finetune-min 0]

1. Candidates come from analyze_student.py (heads / blocks whose removal helped on its 500-item sample).
2. Greedy combination on a separate CONFIRM slice of the dev set (items 600..1800): an edit is kept only if
   the combined score (product chrF + 0.3 x place chrF) rises by > 0.2.
3. The final edited model is checked on the hand-written gold product set (student alone) and must not be worse.
4. Optional short fine-tune with the pruned weights held at zero (gradients masked), then re-check.
Writes <model_dir>_edited/ (student.pt + ONNX with key/value cache) and build/work/edits_<name>.json.
"""
import argparse, json, os, random, sys, copy
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import analyze_student as AS
from student_data import student_input


def gold_items():
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    return [(g["en"], student_input(s)) for g in G for s in g["src"].values()]


def gold_score(model, si, tv, items):
    outs = AS.greedy(model, [s for _, s in items], si, tv)
    ex = sum(AS.clean(o) in [AS.clean(r) for r in refs] for o, (refs, _) in zip(outs, items))
    ch = sum(max(AS.chrf(AS.clean(o), AS.clean(r)) for r in refs) for o, (refs, _) in zip(outs, items)) / len(items)
    return {"gold_exact": round(ex / len(items), 4), "gold_chrF": round(ch, 2)}


def combined(r):
    return r["bt_chrF"] + 0.3 * r.get("gaz_chrF", 0)


def edit_fn(model, what):
    """what: 'dec1.self.head0' | 'enc0.ffn' | 'dec2.whole' -> returns list of (param, column slice or None)."""
    stack, kind, *rest = what.split(".")
    blk = (model.enc if stack.startswith("enc") else model.dec)[int(stack[3:])]
    if kind in ("self", "cross"):
        mha = blk.sa if kind == "self" else blk.ca; h = int(rest[0][4:]); hd = mha.hd
        return [(mha.o.weight, slice(h * hd, (h + 1) * hd))]
    if kind == "ffn":
        return [(blk.ff[2].weight, None), (blk.ff[2].bias, None)]
    return [(blk.sa.o.weight, None), (blk.sa.o.bias, None), (blk.ff[2].weight, None), (blk.ff[2].bias, None)] + \
        ([(blk.ca.o.weight, None), (blk.ca.o.bias, None)] if blk.cross else [])


def apply(model, edits):
    with torch.no_grad():
        for w in edits:
            for p, sl in edit_fn(model, w):
                if sl is None: p.zero_()
                else: p[:, sl] = 0


def finetune(model, edits, train, si, ti, minutes, lr=1e-4):
    import time, math, torch.nn as nn
    from train_student import PAD, UNK, BOS, EOS, MAX_S, MAX_T
    masks = []
    for w in edits:
        for p, sl in edit_fn(model, w):
            m = torch.ones_like(p)
            if sl is None: m.zero_()
            else: m[:, sl] = 0
            masks.append((p, m))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    if AS.TOK:                                     # v5 word-piece model
        import bpe as BPE
        ts = lambda x: BPE.encode(x, AS.TOK["src"]); tt = lambda x: BPE.encode(x, AS.TOK["tgt"])
    else:
        ts = tt = lambda x: x
    data = [([si.get(c, UNK) for c in ts(p["src"])[:model.sp.num_embeddings]], [BOS] + [ti.get(c, UNK) for c in tt(p["tgt"])[:model.tp.num_embeddings - 2]] + [EOS]) for p in train]
    rng = random.Random(1); t_end = time.time() + minutes * 60; model.train(); steps = 0
    while time.time() < t_end:
        b = rng.sample(data, 64); S = max(len(x[0]) for x in b); T = max(len(x[1]) for x in b)
        src = torch.tensor([x[0] + [PAD] * (S - len(x[0])) for x in b]); tgt = torch.tensor([x[1] + [PAD] * (T - len(x[1])) for x in b])
        lg = model(src, tgt[:, :-1]); loss = lossf(lg.reshape(-1, lg.shape[-1]), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        for p, m in masks:
            if p.grad is not None: p.grad *= m
        opt.step(); steps += 1
        with torch.no_grad():
            for p, m in masks: p *= m
    model.eval(); return steps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("--dev", default="pairs_dev.jsonl")
    ap.add_argument("--train", default="pairs_train.jsonl"); ap.add_argument("--finetune-min", type=float, default=0)
    ap.add_argument("--analysis", default=None); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    md = os.path.normpath(a.model_dir); name = os.path.basename(md)
    model, si, tv, ck = AS.load(md); ti = {c: i for i, c in enumerate(tv)}
    an = json.load(open(a.analysis or os.path.join(HERE, "work", f"analysis_{name}.json")))
    dev = [json.loads(l) for l in open(os.path.join(HERE, "work", a.dev), encoding="utf-8")]; random.Random(5).shuffle(dev)
    bt = [p for p in dev if p["kind"] == "bt"]; gz = [p for p in dev if p["kind"] == "gaz"]
    lo = 600 if len(bt) >= 900 else 350                         # analysis used bt[:350] / bt[:600]
    confirm = bt[lo:1500] + gz[200:500]                       # not the slice the analysis used
    gold = gold_items(); base_sd = copy.deepcopy(model.state_dict())
    rep = {"model": name, "base": {**AS.score(model, confirm, si, tv), **gold_score(model, si, tv, gold)}, "tried": []}
    print("base", rep["base"], flush=True)
    cands = sorted([x for x in an["ablations"] if x["d_bt"] + 0.3 * x["d_gaz"] > 0.2], key=lambda x: -(x["d_bt"] + 0.3 * x["d_gaz"]))
    kept, best = [], combined(rep["base"])
    for c in cands:
        model.load_state_dict(base_sd); apply(model, kept + [c["what"]]); r = AS.score(model, confirm, si, tv)
        ok = combined(r) > best + 0.2
        rep["tried"].append({"edit": c["what"], "with": list(kept), **r, "kept": ok}); print(c["what"], r, "KEEP" if ok else "drop", flush=True)
        if ok:
            kept.append(c["what"]); best = combined(r)
    model.load_state_dict(base_sd); apply(model, kept)
    final = {**AS.score(model, confirm, si, tv), **gold_score(model, si, tv, gold)}
    rep["kept"] = kept; rep["edited"] = final; print("edited", kept, final, flush=True)
    if a.finetune_min > 0:
        train = [json.loads(l) for l in open(os.path.join(HERE, "work", a.train), encoding="utf-8")]
        steps = finetune(model, kept, train, si, ti, a.finetune_min)
        ft = {**AS.score(model, confirm, si, tv), **gold_score(model, si, tv, gold)}
        rep["finetuned"] = {**ft, "steps": steps, "minutes": a.finetune_min}; print("fine-tuned", ft, flush=True)
        if combined(ft) + ft["gold_chrF"] * 0.5 < combined(final) + final["gold_chrF"] * 0.5:
            model.load_state_dict(base_sd); apply(model, kept); rep["finetuned"]["used"] = False
        else:
            final = ft; rep["finetuned"]["used"] = True
    better = combined(final) > combined(rep["base"]) + 0.2 and final["gold_chrF"] >= rep["base"]["gold_chrF"] - 0.3
    rep["decision"] = "use edited model" if better else "keep original weights"
    print("DECISION:", rep["decision"], flush=True)
    json.dump(rep, open(os.path.join(HERE, "work", f"edits_{name}.json"), "w"), indent=1)
    if better:
        import train_student2 as T2
        out = md + "_edited"; os.makedirs(out, exist_ok=True)
        torch.save({"model": model.state_dict(), "svocab": ck["svocab"], "tvocab": ck["tvocab"], "cfg": ck.get("cfg", {}), "edits": kept},
                   os.path.join(out, "student.pt"))
        T2.export(model, ck["svocab"], ck["tvocab"], out)
        if AS.TOK:                                 # keep the word-piece merges with the edited model
            vp = os.path.join(out, "vocab.json"); v = json.load(open(vp, encoding="utf-8")); v["bpe"] = AS.TOK
            json.dump(v, open(vp, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
