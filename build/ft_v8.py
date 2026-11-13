"""v8: the fastest model that keeps v6/v7 quality.
  encoder   v6's 6 layers (runs once per label) + one small expert per language (encoder only: no per-token cost)
  decoder   2 layers taken from v6's 6 (runs once per output token, so this is where the time goes)
  output    English-only vocabulary (pieces that occur in English targets) instead of the shared 39k
  data      everything v6 saw + short labels written by Hy-MT2-1.8B for ko / th / vi / ja / zh

  python build/ft_v8.py prep                      vocabularies, starting weights from v6, routes, Hy short-label pairs
  python build/ft_v8.py train --minutes N --total-min T [--resume]
  python build/ft_v8.py eval                      gold + ABO + tuning labels, model alone
Restart-safe like ft_marian.py."""
import argparse, glob, json, math, os, random, re, sys, time
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import sentencepiece as spm
import ft_marian as FT, ft_moe as MOE
from ft_marian import EOS, UNK, PAD
from transformers import MarianConfig, MarianMTModel, GenerationConfig
from huggingface_hub import snapshot_download
LANGS = MOE.LANGS
KIND_W = dict(FT.KIND_W, short_hy=4)
SCRIPT = {"ko": r"[가-힣]", "th": r"[฀-๿]", "ja": r"[぀-ヿ一-鿿]", "zh": r"[一-鿿]"}


def out_dir(tag):
    d = os.path.join(ROOT, "models", tag); os.makedirs(d, exist_ok=True); return d


class Tok:
    """Source and target sides have their own piece lists (ids 0,1,2 = </s>, <unk>, <pad> in both)."""
    def __init__(self, d):
        b = FT.out_dir("marian_ft")
        self.sp_s = spm.SentencePieceProcessor(model_file=os.path.join(b, "source.spm")); self.sp_t = spm.SentencePieceProcessor(model_file=os.path.join(b, "target.spm"))
        v = json.load(open(os.path.join(d, "vocab8.json"), encoding="utf-8")); self.src = v["src"]; self.tgt = v["tgt"]
        self.si = {p: i for i, p in enumerate(self.src)}; self.ti = {p: i for i, p in enumerate(self.tgt)}

    def enc(self, texts, side="src", max_len=64):
        sp, ids = (self.sp_s, self.si) if side == "src" else (self.sp_t, self.ti)
        return [[ids.get(p, UNK) for p in ps[:max_len - 1]] + [EOS] for ps in sp.encode(texts, out_type=str)]

    def dec(self, ids):
        return "".join(self.tgt[i] for i in ids if i not in (EOS, PAD)).replace("▁", " ").strip()


def build(vs, vt, dec_layers, enc_layers=6):
    cfg = MarianConfig.from_pretrained(snapshot_download(FT.BASE))
    cfg.vocab_size = vs; cfg.decoder_vocab_size = vt; cfg.share_encoder_decoder_embeddings = False; cfg.decoder_layers = dec_layers; cfg.encoder_layers = enc_layers
    cfg.pad_token_id = PAD; cfg.eos_token_id = EOS; cfg.decoder_start_token_id = PAD
    for k in ("max_length", "num_beams", "bad_words_ids"):
        cfg.__dict__.pop(k, None)
    m = MarianMTModel(cfg)
    m.lm_head.weight = m.model.decoder.embed_tokens.weight                     # output layer shares the English embeddings
    m.generation_config = GenerationConfig(pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD, bad_words_ids=[[PAD]], num_beams=1)
    return m


class Experts(nn.Module):
    """Encoder-only language experts: h + relu(h @ down) @ up after every encoder layer, chosen per example."""
    def __init__(self, n_layers, d, r):
        super().__init__()
        self.down = nn.Parameter(torch.randn(len(LANGS), n_layers, d, r) * (1.0 / math.sqrt(d)))
        self.up = nn.Parameter(torch.zeros(len(LANGS), n_layers, r, d)); self.idx = None

    def apply(self, h, layer):
        i = self.idx; on = (i > 0).to(h.dtype)[:, None, None]
        return h + on * torch.bmm(torch.relu(torch.bmm(h, self.down[i, layer])), self.up[i, layer])


def attach(m, ex):
    for k, layer in enumerate(m.model.encoder.layers):
        def hook(mod, inp, out, k=k):
            return (ex.apply(out[0], k),) + tuple(out[1:]) if isinstance(out, tuple) else ex.apply(out, k)
        layer.register_forward_hook(hook)


def hy_pairs():
    """Hy-MT2 short labels -> pairs (kind short_hy): right script, not a copy, sane length."""
    from student_data import student_input
    p = os.path.join(W, "bt_hy_q.jsonl"); out = []; seen = set()
    for l in open(p, encoding="utf-8") if os.path.exists(p) else []:
        r = json.loads(l); src = student_input(r["src"]); en = r["en"].lower()
        if not src or src == en or (src, en) in seen or not (0.2 <= len(src) / max(1, len(en)) <= 3.5):
            continue
        if r["lang"] in SCRIPT and not re.search(SCRIPT[r["lang"]], src):
            continue
        if r["lang"] == "vi" and src.isascii():
            continue                                             # Vietnamese without any diacritic: most likely left in English
        seen.add((src, en)); out.append({"kind": "short_hy", "lang": r["lang"], "src": src, "tgt": en})
    return out


def all_train():
    t = FT.read_pairs("MAX", "train") + [json.loads(l) for l in open(os.path.join(W, "pairs_Q_train.jsonl"), encoding="utf-8")]
    return t + [json.loads(l) for l in open(os.path.join(W, "pairs_HQ_train.jsonl"), encoding="utf-8")]


def prep(a):
    d = out_dir(a.tag); b = FT.out_dir("marian_ft")
    hq = hy_pairs()
    with open(os.path.join(W, "pairs_HQ_train.jsonl"), "w", encoding="utf-8") as f:
        for p in hq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print("Hy short-label pairs", len(hq), dict(Counter(p["lang"] for p in hq)), flush=True)
    v6 = json.load(open(os.path.join(b, "vocab.json"), encoding="utf-8")); old = {p: i for i, p in enumerate(v6)}
    ss = spm.SentencePieceProcessor(model_file=os.path.join(b, "source.spm")); st = spm.SentencePieceProcessor(model_file=os.path.join(b, "target.spm"))
    train = all_train(); dev = FT.read_pairs("MAX", "dev")
    S = [p["src"] for p in train + dev]; T = [p["tgt"] for p in train + dev]
    try:
        from bt_v2 import titles
        T += [t.lower() for t in titles()] + [t.lower() for t in json.load(open(os.path.join(W, "abo_titles.json"), encoding="utf-8"))]
    except Exception as e:
        print("extra titles skipped:", e)
    cs, ct = Counter(), Counter()
    for i in range(0, len(S), 5000):
        for ps in ss.encode(S[i:i + 5000], out_type=str):
            cs.update(ps)
    for i in range(0, len(T), 5000):
        for ps in st.encode(T[i:i + 5000], out_type=str):
            ct.update(ps)
    spec = ["</s>", "<unk>", "<pad>"]
    src = spec + [p for p in v6[3:] if cs[p] >= 1 or len(p.replace("▁", "")) <= 1]
    tgt = spec + [p for p in v6[3:] if ct[p] >= 1]
    json.dump({"src": src, "tgt": tgt}, open(os.path.join(d, "vocab8.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("source pieces", len(src), "| English output pieces", len(tgt), "| v6 shared", len(v6), flush=True)
    _, m6, _ = FT.load(b, os.path.join(ROOT, "models", "student_V6", "student.pt")); sd6 = m6.state_dict()
    keep = [int(x) for x in a.dec_keep.split(",")]; m = build(len(src), len(tgt), len(keep)); sd = m.state_dict(); E = sd6["model.shared.weight"]
    for k in sd:
        if k in ("model.shared.weight", "model.encoder.embed_tokens.weight"):
            sd[k] = E[[old[p] for p in src]]
        elif k in ("model.decoder.embed_tokens.weight", "lm_head.weight"):
            sd[k] = E[[old[p] for p in tgt]]
        elif k == "final_logits_bias":
            sd[k] = sd6[k][:, [old[p] for p in tgt]]
        elif k.startswith("model.decoder.layers."):
            j = int(k.split(".")[3]); sd[k] = sd6[k.replace(f"layers.{j}.", f"layers.{keep[j]}.", 1)]
        else:
            sd[k] = sd6[k]
    m.load_state_dict(sd); torch.save({"model": m.state_dict(), "dec_keep": keep}, os.path.join(d, "base.pt"))
    print("params", round(sum(p.numel() for p in set(m.parameters())) / 1e6, 2), "M | decoder layers kept from v6:", keep, flush=True)
    R = json.load(open(os.path.join(W, "moe_routes.json"), encoding="utf-8")); n0 = len(R)
    for p in hq:
        if p["src"] not in R:
            R[p["src"]] = MOE.route(p["src"])
    json.dump(R, open(os.path.join(W, "moe_routes.json"), "w", encoding="utf-8"), ensure_ascii=False); print("routes", n0, "->", len(R), flush=True)


def prep_lite(a):
    """The fast model: cut from the trained complete model (--from tag): fewer encoder layers, one decoder layer,
    no language experts. Same vocabularies."""
    d = out_dir(a.tag); src_d = out_dir(a.from_tag)
    json.dump(json.load(open(os.path.join(src_d, "vocab8.json"), encoding="utf-8")), open(os.path.join(d, "vocab8.json"), "w", encoding="utf-8"), ensure_ascii=False)
    tok = Tok(d); sd8 = torch.load(os.path.join(src_d, "latest.pt"), map_location="cpu")["model"]
    ek = [int(x) for x in a.enc_keep.split(",")]; dk = [int(x) for x in a.dec_keep.split(",")]
    m = build(len(tok.src), len(tok.tgt), len(dk), len(ek)); sd = m.state_dict()
    for k in sd:
        if k.startswith("model.encoder.layers."):
            j = int(k.split(".")[3]); sd[k] = sd8[k.replace(f"layers.{j}.", f"layers.{ek[j]}.", 1)]
        elif k.startswith("model.decoder.layers."):
            j = int(k.split(".")[3]); sd[k] = sd8[k.replace(f"layers.{j}.", f"layers.{dk[j]}.", 1)]
        else:
            sd[k] = sd8[k]
    m.load_state_dict(sd); torch.save({"model": m.state_dict(), "dec_keep": dk, "enc_layers": len(ek), "enc_keep": ek, "lite": True}, os.path.join(d, "base.pt"))
    print("lite params", round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M | encoder layers", ek, "decoder layers", dk, flush=True)


def load(a, ckpt=None):
    d = out_dir(a.tag); tok = Tok(d); base = torch.load(os.path.join(d, "base.pt"), map_location="cpu")
    if base.get("lite"):
        global NOEX
        NOEX = True
    m = build(len(tok.src), len(tok.tgt), len(base["dec_keep"]), base.get("enc_layers", 6)); ex = Experts(len(m.model.encoder.layers), m.config.d_model, a.rank); attach(m, ex)
    sd = torch.load(ckpt, map_location="cpu") if ckpt else base
    m.load_state_dict(sd["model"])
    if "experts" in sd:
        ex.load_state_dict(sd["experts"])
    return tok, m, ex, sd


NOEX = False        # fast model: no language experts (every label takes route 0)


def rt(s, R=None):
    if NOEX:
        return 0
    return R.get(s, 0) if R is not None else MOE.route(s)


def translate(tok, m, ex, texts, routes, bs=48, max_new=48):
    m.eval(); out = [None] * len(texts); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; ids = tok.enc([texts[j] for j in idx]); L = max(map(len, ids))
        x = torch.tensor([s + [PAD] * (L - len(s)) for s in ids]); ex.idx = torch.tensor([routes[j] for j in idx])
        with torch.no_grad():
            g = m.generate(input_ids=x, attention_mask=(x != PAD).long(), num_beams=1, do_sample=False, max_new_tokens=max_new)
        for j, r in zip(idx, g.tolist()):
            r = r[1:]; r = r[:r.index(EOS)] if EOS in r else r
            out[j] = tok.dec(r)
    return out


def batches(train, R, tok, rng, max_tok):
    w = [KIND_W.get(p["kind"], 1) for p in train]
    while True:
        chunk = rng.choices(train, weights=w, k=3072)
        xs = tok.enc([p["src"] for p in chunk]); ys = tok.enc([p["tgt"] for p in chunk], "tgt", 48); rs = [rt(p["src"], R) for p in chunk]
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
            yield (torch.tensor([xs[i] + [PAD] * (X - len(xs[i])) for i in b]), torch.tensor([ys[i] + [PAD] * (Y - len(ys[i])) for i in b]),
                   torch.tensor([rs[i] for i in b]))


def score(outs, refs):
    n = torch.get_num_threads(); r = FT.score(outs, refs); torch.set_num_threads(n); return r


def dev_eval(tok, m, ex, dev, R, DL):
    from student_data import student_input
    it = [p for p in dev if p["kind"] == "bt"][:300]; srcs = [p["src"] for p in it]
    r = {"titles": score(translate(tok, m, ex, srcs, [rt(s, R) for s in srcs]), [[p["tgt"]] for p in it])["chrF"]}
    ls = [student_input(s) for g in DL for s in g["src"].values()]
    r["labels"] = score(translate(tok, m, ex, ls, [rt(s) for s in ls]), [g["en"] for g in DL for _ in g["src"]])["chrF"]
    return r


def train(a):
    torch.set_num_threads(a.threads); d = out_dir(a.tag); rng = random.Random(13)
    latest = os.path.join(d, "latest.pt"); resume = a.resume and os.path.exists(latest)
    tok, m, ex, sd = load(a, latest if resume else None)
    base_params = list({id(p): p for p in m.parameters()}.values())
    opt = torch.optim.AdamW([{"params": base_params, "scale": 1.0}, {"params": list(ex.parameters()), "scale": a.ex_lr / a.lr, "weight_decay": 0.0}],
                            lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    step = 0; done = 0.0; logp = os.path.join(d, "train_log.json")
    log = json.load(open(logp)) if resume and os.path.exists(logp) else {"evals": [], "args": vars(a)}
    if resume:
        opt.load_state_dict(sd["opt"]); step = sd["step"]; done = sd["minutes"]; rng.setstate(sd["rng"])
    total = a.total_min or (done + a.minutes); budget = min(a.minutes, total - done)
    if budget <= 0:
        print("nothing left to train"); return
    R = json.load(open(os.path.join(W, "moe_routes.json"), encoding="utf-8")); train_p = all_train()
    dev = FT.read_pairs("MAX", "dev"); random.Random(5).shuffle(dev)
    DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
    print(f"pairs {len(train_p)} | step {step} | done {done:.0f} min | this run {budget:.0f} of {total:.0f}", flush=True)
    if step == 0:
        r = {**dev_eval(tok, m, ex, dev, R, DL), "step": 0}; log["evals"].append(r); print("dev", r, flush=True)
    it = batches(train_p, R, tok, rng, a.max_tok); t0 = time.time(); last_save = last_eval = t0; ema = None

    def save(keep=False):
        tmp = latest + ".tmp"
        torch.save({"model": m.state_dict(), "experts": ex.state_dict(), "opt": opt.state_dict(), "step": step, "minutes": done + (time.time() - t0) / 60, "rng": rng.getstate()}, tmp)
        os.replace(tmp, latest); json.dump(log, open(logp, "w"), indent=1)
        if keep:
            torch.save({"model": m.state_dict(), "experts": ex.state_dict(), "step": step}, os.path.join(d, f"step_{step:06d}.pt"))
            for old in sorted(glob.glob(os.path.join(d, "step_*.pt")))[:-3]:
                os.remove(old)

    while True:
        el = (time.time() - t0) / 60
        if el >= budget:
            break
        frac = min(1.0, (done + el) / total); lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr * g["scale"]
        x, y, ri = next(it); m.train(); ex.idx = ri
        dec_in = torch.cat([torch.full((y.shape[0], 1), PAD), y[:, :-1]], 1)
        logits = m(input_ids=x, attention_mask=(x != PAD).long(), decoder_input_ids=dec_in).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=PAD, label_smoothing=0.1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(base_params + list(ex.parameters()), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step} loss {ema:.3f} lr {lr:.2e} {budget - el:.0f} min left", flush=True)
        now = time.time()
        if now - last_save > 300:
            save(); last_save = now
        if now - last_eval > a.ckpt_min * 60:
            r = {**dev_eval(tok, m, ex, dev, R, DL), "step": step}; log["evals"].append(r); print("dev", r, flush=True); save(True); last_eval = last_save = time.time()
    r = {**dev_eval(tok, m, ex, dev, R, DL), "step": step}; log["evals"].append(r); print("dev", r, flush=True); save(True); print("run done", flush=True)


def evaluate(a):
    torch.set_num_threads(a.threads); d = out_dir(a.tag); tok, m, ex, _ = load(a, a.ckpt or os.path.join(d, "latest.pt"))
    from student_data import student_input
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
    sets = {"dev_labels": [(g["en"], s, l) for g in DL for l, s in g["src"].items()], "gold": [(g["en"], s, l) for g in G for l, s in g["src"].items()],
            "abo": [([p["en"]], p["src"], p["lang"]) for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8"))]}
    res = {}
    for name, items in sets.items():
        srcs = [student_input(s) for _, s, _ in items]; rts = [rt(s) for s in srcs]; refs = [r for r, _, _ in items]
        outs = translate(tok, m, ex, srcs, rts); res[name] = {"experts": score(outs, refs), "no_experts": score(translate(tok, m, ex, srcs, [0] * len(srcs)), refs)}
        by = {}
        from analyze_student import clean, chrf
        for o, (rf, _, l) in zip(outs, items):
            by.setdefault(l, []).append(max(chrf(clean(o), clean(x)) for x in rf))
        res[name]["by_lang"] = {l: round(sum(v) / len(v), 1) for l, v in by.items()}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(W, f"v8_eval_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--tag", default="v8"); ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--dec-keep", default="0,5"); ap.add_argument("--enc-keep", default="0,2,5"); ap.add_argument("--from-tag", default="v8"); ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--total-min", type=float, default=0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--ex-lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-tok", type=int, default=1400); ap.add_argument("--ckpt-min", type=float, default=30); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(); {"prep": prep, "prep_lite": prep_lite, "train": train, "eval": evaluate}[a.mode](a)
