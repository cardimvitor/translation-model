"""v7: language experts on top of v6 (mixture of experts with hard routing).
The v6 translator is frozen. Each language gets its own small expert: one adapter (d -> r -> d, added to the layer
output) in every encoder and decoder layer. The language detector of the Fast model (runtime/normalizer.detect)
picks the expert; training uses the same detector, so routing is identical at training time and on the phone.
Only the chosen expert's parameters are used for a label; route 0 ("none": unsure / English) is plain v6.

  python build/ft_moe.py routes                      detector route of every training / dev source -> work/moe_routes.json
  python build/ft_moe.py train --minutes N --total-min T [--resume]
  python build/ft_moe.py eval                        gold + ABO, model alone, with and without experts"""
import argparse, glob, json, math, os, random, sys, time
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_marian as FT
from ft_marian import EOS, UNK, PAD
LANGS = ["none", "zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"]
MIN_CONF = 0.5


def route(text):
    """Expert index for a label: the detector's first language if it is one of ours and confident enough."""
    import normalizer as F
    r = F.process("translate", text)["language"]
    if r and r[0][0] in LANGS and r[0][1] >= MIN_CONF:
        return LANGS.index(r[0][0])
    return 0


class Experts(nn.Module):
    def __init__(self, n_layers, d, r):
        super().__init__()
        self.down = nn.Parameter(torch.randn(len(LANGS), n_layers, d, r) * (1.0 / math.sqrt(d)))
        self.up = nn.Parameter(torch.zeros(len(LANGS), n_layers, r, d))          # zero: the model starts as plain v6
        self.idx = None                                                            # (B,) route per example, set before each forward

    def apply(self, h, layer):
        i = self.idx; on = (i > 0).to(h.dtype)[:, None, None]
        z = torch.relu(torch.bmm(h, self.down[i, layer])); return h + on * torch.bmm(z, self.up[i, layer])


def attach(m, ex):
    """Forward hooks: add the routed expert to the output of every encoder and decoder layer."""
    layers = list(m.model.encoder.layers) + list(m.model.decoder.layers)
    for k, layer in enumerate(layers):
        def hook(mod, inp, out, k=k):
            if isinstance(out, tuple):
                return (ex.apply(out[0], k),) + tuple(out[1:])
            return ex.apply(out, k)
        layer.register_forward_hook(hook)
    return len(layers)


def load(a, ckpt=None):
    d = FT.out_dir("marian_ft"); tok, m, _ = FT.load(d, os.path.join(ROOT, "models", "student_V6", "student.pt"))
    for p in m.parameters():
        p.requires_grad_(False)
    n = len(m.model.encoder.layers) + len(m.model.decoder.layers); ex = Experts(n, m.config.d_model, a.rank); attach(m, ex)
    sd = None
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu"); ex.load_state_dict(sd["experts"])
    return tok, m, ex, sd


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


def routes_cmd(a):
    R = {}
    for name in ("pairs_MAX_train.jsonl", "pairs_MAX_dev.jsonl", "pairs_Q_train.jsonl"):
        for l in open(os.path.join(W, name), encoding="utf-8"):
            s = json.loads(l)["src"]
            if s not in R:
                R[s] = route(s)
    json.dump(R, open(os.path.join(W, "moe_routes.json"), "w", encoding="utf-8"), ensure_ascii=False)
    from collections import Counter
    print({LANGS[k]: v for k, v in sorted(Counter(R.values()).items())})


def batches(train, R, tok, rng, max_tok):
    w = [FT.KIND_W.get(p["kind"], 1) for p in train]
    while True:
        chunk = rng.choices(train, weights=w, k=3072)
        xs = tok.enc([p["src"] for p in chunk]); ys = tok.enc([p["tgt"] for p in chunk], "tgt", 48); rs = [R.get(p["src"], 0) for p in chunk]
        order = sorted(range(len(chunk)), key=lambda i: (len(xs[i]), len(ys[i]))); bl = []; cur = []; mx = my = 0
        for i in order:
            if rs[i] == 0:
                continue                                        # route "none" has no expert to train
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


def dev_eval(tok, m, ex, dev, R):
    r = {}
    it = [p for p in dev if p["kind"] == "bt"][:400]; srcs = [p["src"] for p in it]; refs = [[p["tgt"]] for p in it]
    r["bt_experts"] = FT.score(translate(tok, m, ex, srcs, [R.get(s, 0) for s in srcs]), refs)["chrF"]
    torch.set_num_threads(4)
    return r


def train(a):
    torch.set_num_threads(a.threads); d = FT.out_dir(a.tag); rng = random.Random(11)
    latest = os.path.join(d, "latest.pt"); resume = a.resume and os.path.exists(latest)
    tok, m, ex, sd = load(a, latest if resume else None)
    opt = torch.optim.AdamW(ex.parameters(), lr=a.lr, weight_decay=0.0); step = 0; done = 0.0
    logp = os.path.join(d, "train_log.json"); log = json.load(open(logp)) if resume and os.path.exists(logp) else {"evals": [], "args": vars(a)}
    if resume:
        opt.load_state_dict(sd["opt"]); step = sd["step"]; done = sd["minutes"]; rng.setstate(sd["rng"])
    total = a.total_min or (done + a.minutes); budget = min(a.minutes, total - done)
    if budget <= 0:
        print("nothing left to train"); return
    R = json.load(open(os.path.join(W, "moe_routes.json"), encoding="utf-8"))
    train_p = FT.read_pairs("MAX", "train") + [json.loads(l) for l in open(os.path.join(W, "pairs_Q_train.jsonl"), encoding="utf-8")]
    train_p = [p for p in train_p if p["kind"] != "gaz"]                      # place names stay with Fast
    dev = FT.read_pairs("MAX", "dev"); random.Random(5).shuffle(dev)
    print(f"pairs {len(train_p)} | step {step} | done {done:.0f} min | this run {budget:.0f} of {total:.0f} | expert params per language {ex.down[0].numel() * 2}", flush=True)
    if step == 0:
        r = {**dev_eval(tok, m, ex, dev, R), "step": 0}; log["evals"].append(r); print("dev", r, flush=True)
    it = batches(train_p, R, tok, rng, a.max_tok); t0 = time.time(); last_save = last_eval = t0; ema = None

    def save():
        tmp = latest + ".tmp"
        torch.save({"experts": ex.state_dict(), "opt": opt.state_dict(), "step": step, "minutes": done + (time.time() - t0) / 60, "rng": rng.getstate()}, tmp)
        os.replace(tmp, latest); json.dump(log, open(logp, "w"), indent=1)

    while True:
        el = (time.time() - t0) / 60
        if el >= budget:
            break
        frac = min(1.0, (done + el) / total); lr = a.lr * min(1.0, (step + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, ri = next(it); m.train(); ex.idx = ri          # dropout on, as in the v6 fine-tune
        dec_in = torch.cat([torch.full((y.shape[0], 1), PAD), y[:, :-1]], 1)
        logits = m(input_ids=x, attention_mask=(x != PAD).long(), decoder_input_ids=dec_in).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=PAD, label_smoothing=0.1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step} loss {ema:.3f} lr {lr:.2e} {budget - el:.0f} min left", flush=True)
        now = time.time()
        if now - last_save > 300:
            save(); last_save = now
        if now - last_eval > a.ckpt_min * 60:
            r = {**dev_eval(tok, m, ex, dev, R), "step": step}; log["evals"].append(r); print("dev", r, flush=True); save(); last_eval = last_save = time.time()
    r = {**dev_eval(tok, m, ex, dev, R), "step": step}; log["evals"].append(r); print("dev", r, flush=True); save(); print("run done", flush=True)


def evaluate(a):
    torch.set_num_threads(a.threads); d = FT.out_dir(a.tag); tok, m, ex, _ = load(a, os.path.join(d, "latest.pt"))
    from student_data import student_input
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
    sets = {"dev_labels": [(g["en"], s, l) for g in DL for l, s in g["src"].items()], "gold": [(g["en"], s, l) for g in G for l, s in g["src"].items()],
            "abo": [([p["en"]], p["src"], p["lang"]) for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8"))]}
    res = {}
    for name, items in sets.items():
        srcs = [student_input(s) for _, s, _ in items]; rt = [route(s) for _, s, _ in items]; refs = [r for r, _, _ in items]
        res[name] = {"experts": FT.score(translate(tok, m, ex, srcs, rt), refs), "plain_v6": FT.score(translate(tok, m, ex, srcs, [0] * len(srcs)), refs),
                     "routed": round(sum(r > 0 for r in rt) / len(rt), 3), "route_is_true_language": round(sum(LANGS[r] == l for r, (_, _, l) in zip(rt, items)) / len(rt), 3)}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(W, f"moe_eval_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--tag", default="moe_v7"); ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--total-min", type=float, default=0); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-tok", type=int, default=1400); ap.add_argument("--ckpt-min", type=float, default=30); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); {"routes": routes_cmd, "train": train, "eval": evaluate}[a.mode](a)
