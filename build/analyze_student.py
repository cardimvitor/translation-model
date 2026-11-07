"""Weight analysis of a trained student (does editing the weights help?).

For a fixed held-out sample (product titles + place names) it measures, with batched greedy decoding in PyTorch:
  - baseline (fp32 PyTorch) and the shipped int8 ONNX;
  - ablations: each attention head (encoder self, decoder self, decoder cross), each FFN block, each whole layer zeroed;
  - embedding health: <unk> rate, unused rows, near-duplicate rows.
Writes build/work/analysis_<name>.json.  Usage: python analyze_student.py [model_dir] [n_items]
"""
import json, math, os, re, sys, time, random
from collections import Counter
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import train_student as TS
torch.set_num_threads(int(os.environ.get("THREADS", 1)))


def clean(s):
    s = s.lower().strip(); s = re.sub(r"\s+([.,!?])", r"\1", s); s = re.sub(r"[.!?]+$", "", s).strip()
    s = re.sub(r"^(the|a|an|-)\s+", "", s); return re.sub(r"\s+", " ", s)


def chrf(hyp, ref, n=6, beta=2):
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", ""); P = R = 0; k = 0
    for i in range(1, n + 1):
        h = Counter(hyp[j:j + i] for j in range(len(hyp) - i + 1)); r = Counter(ref[j:j + i] for j in range(len(ref) - i + 1))
        if not h or not r: continue
        m = sum((h & r).values()); P += m / sum(h.values()); R += m / sum(r.values()); k += 1
    if not k: return 0.0
    P, R = P / k, R / k
    return 0.0 if P + R == 0 else (1 + beta ** 2) * P * R / (beta ** 2 * P + R) * 100


@torch.no_grad()
def greedy(model, srcs, si, tv, bs=64, max_t=64):
    outs = []
    for i in range(0, len(srcs), bs):
        b = srcs[i:i + bs]
        ids = [[si.get(c, TS.UNK) for c in s[:TS.MAX_S]] or [TS.UNK] for s in b]
        S = max(map(len, ids)); src = torch.tensor([x + [TS.PAD] * (S - len(x)) for x in ids])
        mem = model.encode(src); tgt = torch.full((len(b), 1), TS.BOS); done = torch.zeros(len(b), dtype=torch.bool)
        for _ in range(max_t - 1):
            lg = model.decode(tgt, mem, src)[:, -1]; lg[:, [TS.PAD, TS.UNK, TS.BOS]] = -1e9
            nx = lg.argmax(-1); nx[done] = TS.PAD; tgt = torch.cat([tgt, nx[:, None]], 1); done |= nx == TS.EOS
            if done.all(): break
        for row in tgt[:, 1:].tolist():
            outs.append("".join(tv[t] for t in row if t not in (TS.PAD,) and t != TS.EOS).split("</s>")[0] if TS.EOS not in row
                        else "".join(tv[t] for t in row[:row.index(TS.EOS)]))
    return outs


def score(model, items, si, tv):
    outs = greedy(model, [p["src"] for p in items], si, tv); r = {}
    for kind in ("bt", "gaz"):
        idx = [i for i, p in enumerate(items) if p["kind"] == kind]
        if idx:
            r[kind + "_chrF"] = round(sum(chrf(clean(outs[i]), clean(items[i]["tgt"])) for i in idx) / len(idx), 2)
            r[kind + "_exact"] = round(sum(clean(outs[i]) == clean(items[i]["tgt"]) for i in idx) / len(idx), 4)
    return r


def load(md):
    ck = torch.load(os.path.join(md, "student.pt"), map_location="cpu")
    sv, tv = ck["svocab"], ck["tvocab"]; cfg = ck.get("cfg", {})
    m = TS.Student(len(sv), len(tv), **cfg); m.load_state_dict(ck["model"]); m.eval()
    return m, {c: i for i, c in enumerate(sv)}, tv, ck


def main(md, n):
    name = os.path.basename(os.path.normpath(md))
    dev = [json.loads(l) for l in open(os.path.join(HERE, "work", os.environ.get("DEV", "pairs_dev.jsonl")), encoding="utf-8")]
    random.Random(5).shuffle(dev)
    items = [p for p in dev if p["kind"] == "bt"][: int(n * 0.7)] + [p for p in dev if p["kind"] == "gaz"][: int(n * 0.3)]
    model, si, tv, ck = load(md); sd = {k: v.clone() for k, v in model.state_dict().items()}
    t0 = time.time(); base = score(model, items, si, tv); dt = time.time() - t0
    print("baseline", base, f"{dt:.0f}s", flush=True)
    res = {"model": name, "n": len(items), "baseline": base, "ablations": []}
    H = model.enc[0].sa.h; hd = model.enc[0].sa.hd

    def run(label, edit):
        model.load_state_dict(sd); edit(); r = score(model, items, si, tv)
        d = round(r["bt_chrF"] - base["bt_chrF"], 2); dg = round(r.get("gaz_chrF", 0) - base.get("gaz_chrF", 0), 2)
        res["ablations"].append({"what": label, **r, "d_bt": d, "d_gaz": dg}); print(f"{label:28s} bt {d:+6.2f} gaz {dg:+6.2f}", flush=True)

    for stack, blocks in (("enc", model.enc), ("dec", model.dec)):
        for li, b in enumerate(blocks):
            mods = [("self", b.sa)] + ([("cross", b.ca)] if b.cross else [])
            for kind, mha in mods:
                for h in range(H):
                    run(f"{stack}{li}.{kind}.head{h}", lambda mha=mha, h=h: mha.o.weight.data[:, h * hd:(h + 1) * hd].zero_())
            run(f"{stack}{li}.ffn", lambda b=b: (b.ff[2].weight.data.zero_(), b.ff[2].bias.data.zero_()))
            def zero_block(b=b, mods=mods):
                for _, m in mods:
                    m.o.weight.data.zero_(); m.o.bias.data.zero_()
                b.ff[2].weight.data.zero_(); b.ff[2].bias.data.zero_()
            run(f"{stack}{li}.whole", zero_block)
    model.load_state_dict(sd)
    # embedding health
    srcs = Counter(c for p in items for c in p["src"]); unk = sum(n for c, n in srcs.items() if c not in si) / max(1, sum(srcs.values()))
    E = model.se.weight.data[4:]; En = torch.nn.functional.normalize(E, dim=-1); sim = En @ En.T; sim.fill_diagonal_(0)
    res["embedding"] = {"src_vocab": len(si), "dev_unk_char_rate": round(unk, 4),
                        "near_duplicate_rows(cos>0.95)": int((sim > 0.95).sum().item() // 2),
                        "mean_row_norm": round(E.norm(dim=-1).mean().item(), 3)}
    print(res["embedding"], flush=True)
    json.dump(res, open(os.path.join(HERE, "work", f"analysis_{name}.json"), "w"), indent=1)
    helps = [a for a in res["ablations"] if a["d_bt"] > 0.3 and a["d_gaz"] > -0.3]
    print("ablations that HELP:", [(a["what"], a["d_bt"], a["d_gaz"]) for a in helps])
    print("least important (|d|<0.3):", [a["what"] for a in res["ablations"] if abs(a["d_bt"]) < 0.3 and abs(a["d_gaz"]) < 0.3])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "student"), int(sys.argv[2]) if len(sys.argv) > 2 else 500)
