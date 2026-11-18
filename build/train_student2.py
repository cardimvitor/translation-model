"""v4 student training (CPU): per-variant data, checkpoints, dev evaluation, checkpoint averaging, resume,
and an ONNX export with a key/value cache (the decoder runs one token per step on the phone).

  python train_student2.py --variant B --minutes 240                 # new run
  python train_student2.py --variant B --minutes 120 --resume        # train more from the last checkpoint
  python train_student2.py --variant B --export-only                 # average the last checkpoints + export

Data: build/work/pairs_<V>_train.jsonl / pairs_<V>_dev.jsonl (build/student_data2.py).
Output: models/student_<V>/ (encoder_int8.onnx, decoder_int8.onnx, vocab.json, ckpt/*.pt, train_log.json)
"""
import argparse, glob, json, math, os, random, sys, time
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train_student as TS
from train_student import Student, PAD, UNK, BOS, EOS, MAX_S, MAX_T
from analyze_student import greedy, clean, chrf
W = os.path.join(HERE, "work")
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import bpe as BPE
TOK = {"src": None, "tgt": None}          # v5: word-piece merges (--bpe); None = one token per character


def tok(s, side):
    return BPE.encode(s, TOK[side]) if TOK[side] else s


def detok(s):
    return BPE.decode([s]) if TOK["tgt"] else s


def load(v, split):
    return [json.loads(l) for l in open(os.path.join(W, f"pairs_{v}_{split}.jsonl"), encoding="utf-8")]


def dev_score(model, dev, si, tv):
    model.eval()
    with torch.no_grad():
        outs = [detok(o) for o in greedy(model, [tok(p["src"], "src") for p in dev], si, tv)]
    model.train(); r = {}
    for kind in ("bt", "gaz"):
        idx = [i for i, p in enumerate(dev) if p["kind"] == kind]
        if idx:
            r[kind + "_chrF"] = round(sum(chrf(clean(outs[i]), clean(dev[i]["tgt"])) for i in idx) / len(idx), 2)
            r[kind + "_exact"] = round(sum(clean(outs[i]) == clean(dev[i]["tgt"]) for i in idx) / len(idx), 4)
    return r


def batches_by_tokens(data, max_tokens, rng):
    data = sorted(data, key=lambda x: (len(x[0]), len(x[1])))
    out, cur, longest = [], [], 0
    for x in data:
        l = max(len(x[0]), len(x[1]))
        if cur and max(longest, l) * (len(cur) + 1) > max_tokens:
            out.append(cur); cur, longest = [], 0
        cur.append(x); longest = max(longest, l)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--d", type=int, default=320); ap.add_argument("--enc", type=int, default=6); ap.add_argument("--dec", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=1280)
    ap.add_argument("--lr", type=float, default=8e-4); ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--ckpt-min", type=float, default=30); ap.add_argument("--dev-n", type=int, default=600)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--total-min", type=float, default=None); ap.add_argument("--latest-min", type=float, default=5)
    ap.add_argument("--enc-lr-scale", type=float, default=1.0, help="learning-rate factor for the encoder (JEPA-Gen: 0.3)")
    ap.add_argument("--init-enc", default=None, help="JEPA-Gen: start the encoder (and source vocabulary) from models/jepa_*/jepa.pt")
    ap.add_argument("--max-len", type=int, default=MAX_S, help="longest input/output in characters (or pieces)")
    ap.add_argument("--tie", action="store_true", help="share input/output embeddings (needs a joint --bpe)")
    ap.add_argument("--bpe", default=None, help="build/work/bpe_<V>.json (v5 word pieces)")
    ap.add_argument("--avg", type=int, default=3); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0); rng = random.Random(0)
    BP = json.load(open(a.bpe, encoding="utf-8")) if a.bpe else None
    if BP:
        TOK["src"], TOK["tgt"] = BP["src"]["merges"], BP["tgt"]["merges"]
    OUT = os.path.join(ROOT, "models", f"student_{a.variant}"); CK = os.path.join(OUT, "ckpt"); os.makedirs(CK, exist_ok=True)
    logp = os.path.join(OUT, "train_log.json"); log = json.load(open(logp)) if os.path.exists(logp) else {"runs": [], "evals": []}
    dev_all = load(a.variant, "dev"); random.Random(5).shuffle(dev_all)
    dev = [p for p in dev_all if p["kind"] == "bt"][: int(a.dev_n * 0.75)] + [p for p in dev_all if p["kind"] == "gaz"][: int(a.dev_n * 0.25)]
    cks = sorted(glob.glob(os.path.join(CK, "step_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))

    if a.resume or a.export_only:
        src = cks[-1] if cks else None; lp = os.path.join(CK, "latest.pt")
        if os.path.exists(lp) and not a.export_only:
            lt = torch.load(lp, map_location="cpu")
            if src is None or lt["step"] > int(src.split("_")[-1][:-3]):
                src = lp
        ck = lt if src == lp else torch.load(src, map_location="cpu"); sv, tv, cfg = ck["svocab"], ck["tvocab"], ck["cfg"]
        print("resume from", os.path.basename(src), "step", ck["step"], flush=True)
        step0 = ck["step"]
    else:
        train = load(a.variant, "train")
        from collections import Counter
        cs = Counter(c for p in train for c in p["src"]); ct = Counter(c for p in train for c in p["tgt"])
        sv = ["<pad>", "<unk>", "<s>", "</s>"] + [c for c, n in cs.most_common() if n >= 3]
        tv = ["<pad>", "<unk>", "<s>", "</s>"] + [c for c, n in ct.most_common() if n >= 3]
        cfg = dict(d=a.d, enc=a.enc, dec=a.dec, heads=a.heads, ff=a.ff); step0 = 0
        if a.max_len != MAX_S:
            cfg["max_s"] = cfg["max_t"] = a.max_len
        if a.tie:
            cfg["tie"] = True
        if BP:
            sv, tv = BP["src"]["tokens"], BP["tgt"]["tokens"]
        if a.init_enc:
            JE = torch.load(a.init_enc, map_location="cpu"); sv = JE["vocab"]; jc = JE["cfg"]
            cfg.update(d=jc["d"], enc=jc["layers"], heads=jc["heads"], ff=jc["ff"])
    si = {c: i for i, c in enumerate(sv)}; ti = {c: i for i, c in enumerate(tv)}
    model = Student(len(sv), len(tv), **cfg)
    if a.resume or a.export_only:
        model.load_state_dict(ck["model"])
    elif a.init_enc:                                # JEPA context encoder -> student encoder (same parameter names)
        miss = model.load_state_dict({k: v for k, v in JE["E"].items() if not k.startswith("sp.") or v.shape == model.sp.weight.shape}, strict=False)
        print("encoder from JEPA:", a.init_enc, "| not loaded:", len(miss.missing_keys), "decoder/output tensors", flush=True)
    print(f"variant {a.variant} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | cfg {cfg} | src vocab {len(sv)} tgt {len(tv)}", flush=True)

    if not a.export_only:
        train = load(a.variant, "train")
        ML_S, ML_T = cfg.get("max_s", MAX_S), cfg.get("max_t", MAX_T)
        enc_s = lambda s: [si.get(c, UNK) for c in tok(s, "src")[:ML_S]]
        enc_t = lambda s: [BOS] + [ti.get(c, UNK) for c in tok(s, "tgt")[:ML_T - 2]] + [EOS]
        weights = {"bt": 2, "gloss": 2, "gaz": 1, "noise": 1, "kd": 1, "real": 3}
        data = [(enc_s(p["src"]), enc_t(p["tgt"])) for p in train if p["src"] and p["tgt"] for _ in range(weights.get(p["kind"], 1))]
        print("training pairs (weighted)", len(data), flush=True)
        enc_names = ("se.", "sp.", "enc.", "ne.")       # a pretrained (JEPA) encoder learns more gently
        pe = [p for n, p in model.named_parameters() if n.startswith(enc_names)]
        pd = [p for n, p in model.named_parameters() if not n.startswith(enc_names)]
        opt = torch.optim.AdamW([{"params": pe, "scale": a.enc_lr_scale}, {"params": pd, "scale": 1.0}], lr=a.lr, weight_decay=0.01)
        if a.resume and "opt" in ck:
            opt.load_state_dict(ck["opt"])
        lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
        t_start = time.time(); t_end = t_start + a.minutes * 60; step = step0; next_ck = t_start + a.ckpt_min * 60
        done_min = sum(r["minutes"] for r in log["runs"]); total = a.total_min
        if total:   # one schedule across restarts: cosine over total minutes, warmup only at step 0
            warm = 600 if step0 == 0 else 0; peak = a.lr
        else:
            warm = 0 if a.resume else 600; peak = a.lr * (0.6 if a.resume else 1.0)
        losses = []; next_latest = t_start + a.latest_min * 60
        run = {"minutes": 0, "from_step": step0, "to_step": step0, "resume": a.resume}; log["runs"].append(run)
        model.train()
        while time.time() < t_end:
            for b in batches_by_tokens(data, a.tokens, rng):
                now = time.time()
                if now >= t_end:
                    break
                frac = min(1, (done_min + (now - t_start) / 60) / total) if total else (now - t_start) / (a.minutes * 60)
                lr = peak * min(1, (step - step0 + 1) / max(1, warm)) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("scale", 1.0)
                S = max(len(x[0]) for x in b); T = max(len(x[1]) for x in b)
                src = torch.tensor([x[0] + [PAD] * (S - len(x[0])) for x in b]); tgt = torch.tensor([x[1] + [PAD] * (T - len(x[1])) for x in b])
                logits = model(src, tgt[:, :-1]); loss = lossf(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
                losses.append(loss.item())
                if step % 200 == 0:
                    print(f"step {step} loss {sum(losses[-200:]) / len(losses[-200:]):.3f} lr {lr:.2e} {(t_end - now) / 60:.0f} min left", flush=True)
                if now >= next_latest:   # small restart-safe checkpoint
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "svocab": sv, "tvocab": tv, "cfg": cfg, "step": step},
                               os.path.join(CK, "latest.tmp")); os.replace(os.path.join(CK, "latest.tmp"), os.path.join(CK, "latest.pt"))
                    run.update(minutes=round((time.time() - t_start) / 60, 1), to_step=step); json.dump(log, open(logp + ".tmp", "w"), indent=1)
                    os.replace(logp + ".tmp", logp); next_latest = time.time() + a.latest_min * 60
                if now >= next_ck or time.time() >= t_end:
                    save(model, opt, sv, tv, cfg, step, CK); next_ck = time.time() + a.ckpt_min * 60
                    r = dev_score(model, dev, si, tv); r["step"] = step; log["evals"].append(r); print("dev", r, flush=True)
                    run.update(minutes=round((time.time() - t_start) / 60, 1), to_step=step)   # progress survives a restart
                    json.dump(log, open(logp, "w"), indent=1)
        save(model, opt, sv, tv, cfg, step, CK)
        r = dev_score(model, dev, si, tv); r["step"] = step; log["evals"].append(r); print("dev (end)", r, flush=True)
        run.update(minutes=a.minutes, to_step=step)
        json.dump(log, open(logp, "w"), indent=1)

    # checkpoint averaging over the last k checkpoints, keep whichever scores better on dev
    cks = sorted(glob.glob(os.path.join(CK, "step_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    last = torch.load(cks[-1], map_location="cpu"); model.load_state_dict(last["model"]); r_last = dev_score(model, dev, si, tv)
    chosen, r_ch = "last", r_last
    if len(cks) >= 2 and a.avg > 1:
        sds = [torch.load(p, map_location="cpu")["model"] for p in cks[-a.avg:]]
        avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
        model.load_state_dict(avg); r_avg = dev_score(model, dev, si, tv)
        print("last", r_last, "| average of", len(sds), r_avg, flush=True)
        if r_avg["bt_chrF"] + r_avg.get("gaz_chrF", 0) * 0.3 > r_last["bt_chrF"] + r_last.get("gaz_chrF", 0) * 0.3:
            chosen, r_ch = f"avg{len(sds)}", r_avg
        else:
            model.load_state_dict(last["model"])
    log["export"] = {"chosen": chosen, "dev": r_ch, "step": last["step"]}; json.dump(log, open(logp, "w"), indent=1)
    torch.save({"model": model.state_dict(), "svocab": sv, "tvocab": tv, "cfg": cfg}, os.path.join(OUT, "student.pt"))
    export(model, sv, tv, OUT)
    if BP:                                          # the runtimes need the merges to split the input the same way
        vp = os.path.join(OUT, "vocab.json"); v = json.load(open(vp, encoding="utf-8"))
        v["bpe"] = {"src": BP["src"]["merges"], "tgt": BP["tgt"]["merges"]}
        json.dump(v, open(vp, "w", encoding="utf-8"), ensure_ascii=False)


def save(model, opt, sv, tv, cfg, step, CK):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "svocab": sv, "tvocab": tv, "cfg": cfg, "step": step},
               os.path.join(CK, f"step_{step}.pt"))
    cks = sorted(glob.glob(os.path.join(CK, "step_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    for p in cks[:-3]:                      # the last 3 are averaged at export; older ones only take disk
        os.remove(p)


# ------------------------------------------------------------------ ONNX export with a key/value cache
def _heads(x, mha):
    B = x.shape[0]
    return x.view(B, -1, mha.h, mha.hd).transpose(1, 2)


class EncKV(nn.Module):
    """src -> mem, plus every decoder layer's cross-attention keys/values (computed once per input)."""
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, src):
        mem = self.m.encode(src)
        ck = torch.stack([_heads(b.ca.k(mem), b.ca) for b in self.m.dec]); cv = torch.stack([_heads(b.ca.v(mem), b.ca) for b in self.m.dec])
        return ck, cv, self.m.pad_bias(src)


class DecStep(nn.Module):
    """One decoding step: token at position pos, past self-attention keys/values (L,B,h,T,hd) -> logits, new past."""
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, tok, pos, ck, cv, src_bias, pk, pv):
        m = self.m; x = m.te(tok) * m.emb_scale + m.tp(pos); nk, nv = [], []
        for l, b in enumerate(m.dec):
            y = b.n1(x); sa = b.sa
            q = _heads(sa.q(y), sa); k = torch.cat([pk[l], _heads(sa.k(y), sa)], 2); v = torch.cat([pv[l], _heads(sa.v(y), sa)], 2)
            nk.append(k); nv.append(v)
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(sa.hd), dim=-1)
            x = x + sa.o((a @ v).transpose(1, 2).reshape(x.shape[0], -1, sa.h * sa.hd))
            ca = b.ca; q = _heads(ca.q(b.n2(x)), ca)
            a = torch.softmax(q @ ck[l].transpose(-1, -2) / math.sqrt(ca.hd) + src_bias, dim=-1)
            x = x + ca.o((a @ cv[l]).transpose(1, 2).reshape(x.shape[0], -1, ca.h * ca.hd))
            x = x + b.ff(b.n3(x))
        return m.out(m.nd(x))[:, -1, :], torch.stack(nk), torch.stack(nv)


def export(model, sv, tv, OUT):
    model.eval(); L = len(model.dec); h = model.dec[0].sa.h; hd = model.dec[0].sa.hd
    src = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
    enc, dec = EncKV(model).eval(), DecStep(model).eval()
    with torch.no_grad():
        ck, cv, sb = enc(src)
    torch.onnx.export(enc, (src,), os.path.join(OUT, "encoder.onnx"), input_names=["src"], output_names=["ck", "cv", "src_bias"],
                      dynamic_axes={"src": {1: "S"}, "ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}}, opset_version=17, dynamo=False)
    tok = torch.tensor([[BOS]]); pos = torch.tensor([[2]]); pk = torch.zeros(L, 1, h, 2, hd); pv = torch.zeros(L, 1, h, 2, hd)
    torch.onnx.export(dec, (tok, pos, ck, cv, sb, pk, pv), os.path.join(OUT, "decoder.onnx"),
                      input_names=["tok", "pos", "ck", "cv", "src_bias", "pk", "pv"], output_names=["logits", "nk", "nv"],
                      dynamic_axes={"ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}, "pk": {3: "T"}, "pv": {3: "T"}, "nk": {3: "T1"}, "nv": {3: "T1"}},
                      opset_version=17, dynamo=False)
    model.eval()                                   # torch.onnx.export restores the wrappers' (train) mode
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for n in ("encoder", "decoder"):
        quantize_dynamic(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_int8.onnx"), weight_type=QuantType.QInt8)
        os.replace(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_fp32.onnx.bak"))
    json.dump({"src": sv, "tgt": tv, "max_src": model.sp.num_embeddings, "max_tgt": model.tp.num_embeddings, "pad": PAD, "unk": UNK, "bos": BOS, "eos": EOS,
               "kv": True, "layers": L, "heads": h, "head_dim": hd},
              open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".onnx", ".json")):
            print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
