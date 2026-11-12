"""v6: fine-tune a pretrained compact translator (Helsinki-NLP/opus-mt-mul-en, CC BY 4.0) on our product data.

  python build/ft_marian.py prep  [--data MAX]         vocabulary cut to our languages (+ missing Korean pieces),
                                                       pruned starting weights -> models/marian_ft/base.pt
  python build/ft_marian.py train --minutes N --total-min T [--resume]
  python build/ft_marian.py eval  [--ckpt path]        gold labels + ABO real names, model alone

Tokenization is done here with the model's own SentencePiece files and our vocabulary list (models/marian_ft/vocab.json),
so the phone runtime can do exactly the same. Restart-safe: latest.pt every 5 minutes, cosine schedule over --total-min."""
import argparse, glob, json, math, os, random, shutil, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import sentencepiece as spm
from transformers import MarianConfig, MarianMTModel
from huggingface_hub import snapshot_download
BASE = "Helsinki-NLP/opus-mt-mul-en"
KIND_W = {"bt": 2, "gloss": 2, "gaz": 1, "noise": 1, "kd": 1, "real": 3, "short": 3}
EOS, UNK, PAD = 0, 1, 2


def out_dir(tag):
    d = os.path.join(ROOT, "models", tag); os.makedirs(d, exist_ok=True); return d


def read_pairs(data, split):
    return [json.loads(l) for l in open(os.path.join(W, f"pairs_{data}_{split}.jsonl"), encoding="utf-8")]


class Tok:
    def __init__(self, d):
        self.src = spm.SentencePieceProcessor(model_file=os.path.join(d, "source.spm"))
        self.tgt = spm.SentencePieceProcessor(model_file=os.path.join(d, "target.spm"))
        self.vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8")); self.id = {p: i for i, p in enumerate(self.vocab)}

    def enc(self, texts, side="src", max_len=64):
        sp = self.src if side == "src" else self.tgt
        return [[self.id.get(p, UNK) for p in ps[:max_len - 1]] + [EOS] for ps in sp.encode(texts, out_type=str)]

    def dec(self, ids):
        return "".join(self.vocab[i] for i in ids if i not in (EOS, PAD)).replace("▁", " ").strip()


def build(d, V):
    snap = snapshot_download(BASE)
    cfg = MarianConfig.from_pretrained(snap)
    cfg.vocab_size = V; cfg.pad_token_id = PAD; cfg.eos_token_id = EOS; cfg.decoder_start_token_id = PAD
    cfg.decoder_vocab_size = V
    for k in ("max_length", "num_beams", "bad_words_ids"):
        cfg.__dict__.pop(k, None)
    from transformers import GenerationConfig
    m = MarianMTModel(cfg)
    m.generation_config = GenerationConfig(pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD, bad_words_ids=[[PAD]], num_beams=1)
    return m, snap


def prep(a):
    d = out_dir(a.tag); snap = snapshot_download(BASE)
    for f in ("source.spm", "target.spm"):
        shutil.copy(os.path.join(snap, f), os.path.join(d, f))
    old = json.load(open(os.path.join(snap, "vocab.json"), encoding="utf-8"))
    ss = spm.SentencePieceProcessor(model_file=os.path.join(d, "source.spm")); st = spm.SentencePieceProcessor(model_file=os.path.join(d, "target.spm"))
    from collections import Counter
    cs, ct = Counter(), Counter()
    S, T = [], []
    for sp_ in ("train", "dev"):
        for p in read_pairs(a.data, sp_):
            S.append(p["src"]); T.append(p["tgt"])
    for extra in a.extra:                                     # more pair files (same format)
        for l in open(os.path.join(W, extra), encoding="utf-8"):
            p = json.loads(l); S.append(p["src"]); T.append(p["tgt"])
    try:                                                      # every English product title we have, for the target side
        from bt_v2 import titles
        T += [t.lower() for t in titles()]
        T += [t.lower() for t in json.load(open(os.path.join(W, "abo_titles.json"), encoding="utf-8"))]
    except Exception as e:
        print("extra titles skipped:", e)
    for i in range(0, len(S), 5000):
        for ps in ss.encode(S[i:i + 5000], out_type=str):
            cs.update(ps)
    for i in range(0, len(T), 5000):
        for ps in st.encode(T[i:i + 5000], out_type=str):
            ct.update(ps)
    c = cs + ct
    keep = [p for p in old if p not in ("</s>", "<unk>", "<pad>") and (c[p] >= 1 or len(p.replace("▁", "")) <= 1)]
    new = sorted(p for p in c if p not in old and c[p] >= 3)
    vocab = ["</s>", "<unk>", "<pad>"] + keep + new
    json.dump(vocab, open(os.path.join(d, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("vocab", len(vocab), "kept", len(keep), "new pieces", len(new), "e.g.", new[:20], flush=True)
    full = MarianMTModel.from_pretrained(snap); sd = full.state_dict()
    E = sd["model.shared.weight"]; mean = E.mean(0); std = E.std(0)
    g = torch.Generator().manual_seed(1)
    rows = [E[old[p]] if p in old else mean + 0.3 * std * torch.randn(E.shape[1], generator=g) for p in vocab]
    newE = torch.stack(rows); bias = sd["final_logits_bias"][0]
    newB = torch.stack([bias[old[p]] if p in old else bias.mean() for p in vocab])[None]
    m, _ = build(d, len(vocab)); msd = m.state_dict()
    for k in msd:
        if k.endswith("embed_tokens.weight") or k in ("model.shared.weight", "lm_head.weight"):
            msd[k] = newE
        elif k == "final_logits_bias":
            msd[k] = newB
        else:
            msd[k] = sd[k]
    m.load_state_dict(msd)
    assert m.lm_head.weight is m.model.shared.weight or torch.equal(m.lm_head.weight, m.model.shared.weight)
    torch.save({"model": m.state_dict()}, os.path.join(d, "base.pt"))
    print("params", sum(p.numel() for p in m.parameters()) / 1e6, "M", flush=True)


def load(d, ckpt=None):
    tok = Tok(d); m, _ = build(d, len(tok.vocab))
    sd = torch.load(ckpt or os.path.join(d, "base.pt"), map_location="cpu")
    m.load_state_dict(sd["model"]); return tok, m, sd


def translate(tok, m, texts, bs=48, max_new=48):
    m.eval(); out = [None] * len(texts); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; ids = tok.enc([texts[j] for j in idx]); L = max(map(len, ids))
        x = torch.tensor([s + [PAD] * (L - len(s)) for s in ids]); att = (x != PAD).long()
        with torch.no_grad():
            g = m.generate(input_ids=x, attention_mask=att, num_beams=1, do_sample=False, max_new_tokens=max_new)
        for j, r in zip(idx, g.tolist()):
            r = r[1:]; r = r[:r.index(EOS)] if EOS in r else r
            out[j] = tok.dec(r)
    return out


def score(outs, refs):
    n = torch.get_num_threads()
    from analyze_student import clean, chrf      # importing it sets torch to 1 thread: restore
    torch.set_num_threads(n)
    ex = sum(clean(o) in [clean(r) for r in rs] for o, rs in zip(outs, refs)) / len(outs)
    ch = sum(max(chrf(clean(o), clean(r)) for r in rs) for o, rs in zip(outs, refs)) / len(outs)
    return {"exact": round(ex, 4), "chrF": round(ch, 2)}


def dev_eval(tok, m, dev):
    r = {}
    for k, n in (("bt", 300), ("real", 200), ("gaz", 150)):
        it = [p for p in dev if p["kind"] == k][:n]
        if it:
            s = score(translate(tok, m, [p["src"] for p in it]), [[p["tgt"]] for p in it]); r[k + "_chrF"] = s["chrF"]; r[k + "_exact"] = s["exact"]
    return r


def test_eval(tok, m):
    from student_data import student_input
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    gold = [(g["en"], s) for g in G for s in g["src"].values()]
    abo = [([p["en"]], p["src"]) for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8"))]
    res = {}
    for name, items in (("gold", gold), ("abo", abo)):
        t0 = time.time(); outs = translate(tok, m, [student_input(s) for _, s in items])
        res[name] = {**score(outs, [r for r, _ in items]), "ms": round(1000 * (time.time() - t0) / len(items), 1)}
        res[name + "_outs"] = [[s, o] for (_, s), o in zip(items, outs)]
    return res


def batches(train, tok, rng, max_tok):
    w = [KIND_W.get(p["kind"], 1) for p in train]
    while True:
        chunk = rng.choices(train, weights=w, k=3072)
        xs = tok.enc([p["src"] for p in chunk]); ys = tok.enc([p["tgt"] for p in chunk], "tgt", 48)
        order = sorted(range(len(chunk)), key=lambda i: (len(xs[i]), len(ys[i]))); bl = []; cur = []; mx = my = 0
        for i in order:
            nx, ny = max(mx, len(xs[i])), max(my, len(ys[i]))
            if cur and (nx + ny) * (len(cur) + 1) > max_tok:
                bl.append(cur); cur = []; nx, ny = len(xs[i]), len(ys[i])
            cur.append(i); mx, my = nx, ny
        if cur:
            bl.append(cur)
        rng.shuffle(bl)
        for b in bl:
            X = max(len(xs[i]) for i in b); Y = max(len(ys[i]) for i in b)
            x = torch.tensor([xs[i] + [PAD] * (X - len(xs[i])) for i in b]); y = torch.tensor([ys[i] + [PAD] * (Y - len(ys[i])) for i in b])
            yield x, y


def train(a):
    torch.set_num_threads(a.threads); d = out_dir(a.tag); rng = random.Random(7)
    logp = os.path.join(d, "train_log.json"); log = json.load(open(logp)) if os.path.exists(logp) and a.resume else {"runs": [], "evals": [], "args": vars(a)}
    latest = os.path.join(d, "latest.pt"); resume = a.resume and os.path.exists(latest)
    tok, m, sd = load(d, latest if resume else None)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    step = 0; done = 0.0
    if resume:
        opt.load_state_dict(sd["opt"]); step = sd["step"]; done = sd["minutes"]; rng.setstate(sd["rng"])
    total = a.total_min or (done + a.minutes); budget = min(a.minutes, total - done)
    if budget <= 0:
        print("nothing left to train"); return
    train_p = read_pairs(a.data, "train"); dev = read_pairs(a.data, "dev"); random.Random(5).shuffle(dev)
    for extra in a.extra:
        train_p += [json.loads(l) for l in open(os.path.join(W, extra), encoding="utf-8")]
    print(f"train pairs {len(train_p)} | step {step} | done {done:.0f} min | this run {budget:.0f} min of {total:.0f}", flush=True)
    it = batches(train_p, tok, rng, a.max_tok); t0 = time.time(); last_save = last_eval = t0; start_step = step; ema = None

    def save_latest():
        tmp = latest + ".tmp"
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step, "minutes": done + (time.time() - t0) / 60, "rng": rng.getstate()}, tmp)
        os.replace(tmp, latest)
        log["runs"] = [r for r in log["runs"] if r.get("from_step") != start_step] + [{"minutes": round((time.time() - t0) / 60, 1), "from_step": start_step, "to_step": step}]
        json.dump(log, open(logp, "w"), indent=1)

    while True:
        el = (time.time() - t0) / 60
        if el >= budget:
            break
        frac = min(1.0, (done + el) / total)
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(it); m.train()
        dec_in = torch.cat([torch.full((y.shape[0], 1), PAD), y[:, :-1]], 1)
        logits = m(input_ids=x, attention_mask=(x != PAD).long(), decoder_input_ids=dec_in).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=PAD, label_smoothing=0.1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step} loss {ema:.3f} lr {lr:.2e} {budget - el:.0f} min left", flush=True)
        now = time.time()
        if now - last_save > 300:
            save_latest(); last_save = now
        if now - last_eval > a.ckpt_min * 60:
            r = {**dev_eval(tok, m, dev), "step": step}; log["evals"].append(r); print("dev", r, flush=True)
            torch.save({"model": m.state_dict(), "step": step}, os.path.join(d, f"step_{step:06d}.pt"))
            for old in sorted(glob.glob(os.path.join(d, "step_*.pt")))[:-3]:
                os.remove(old)
            save_latest(); last_eval = last_save = time.time()
    r = {**dev_eval(tok, m, dev), "step": step}; log["evals"].append(r); print("dev", r, flush=True)
    torch.save({"model": m.state_dict(), "step": step}, os.path.join(d, f"step_{step:06d}.pt"))
    for old in sorted(glob.glob(os.path.join(d, "step_*.pt")))[:-3]:
        os.remove(old)
    save_latest(); print("run done", flush=True)


def evaluate(a):
    torch.set_num_threads(a.threads); d = out_dir(a.tag)
    tok, m, _ = load(d, a.ckpt or os.path.join(d, "latest.pt")); r = test_eval(tok, m)
    name = os.path.basename(a.ckpt or "latest.pt").replace(".pt", "")
    json.dump(r, open(os.path.join(W, f"marian_ft_{a.tag}_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("gold", {k: v for k, v in r["gold"].items()}, "abo", r["abo"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--data", default="MAX"); ap.add_argument("--tag", default="marian_ft")
    ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--total-min", type=float, default=0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--max-tok", type=int, default=1400)
    ap.add_argument("--ckpt-min", type=float, default=30); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(); {"prep": prep, "train": train, "eval": evaluate}[a.mode](a)
