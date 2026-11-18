"""JEPA (Joint-Embedding Predictive Architecture) for product / place labels, character level, CPU.

  python jepa.py pretrain --data MAX --minutes 180 [--ema 0.996] [--mask 0.3] [--vic 0.1] [--pred-layers 2]
  python jepa.py lookup   --data MAX                 # JEPA-Lookup: nearest English name in a bank of known names
Output: models/jepa_<tag>/jepa.pt (+ jepa_log.json)

Model
  context encoder  E  : characters -> states (same layout as the student encoder, so JEPA-Gen can start from it)
  target encoder   E' : exponential moving average of E, never trained by gradients (stop-gradient)
  predictor        P  : small transformer; reads E's states plus query slots and predicts E' states
Objectives (both in embedding space, no characters are predicted):
  cross-lingual  P(E(foreign label), [global query])  ~  mean-pool(E'(English title))       (cosine)
  masked spans   P(E(text with spans masked), [queries at masked positions]) ~ E'(text) there (smooth L1)
Collapse is avoided by the EMA target + the predictor asymmetry, and optionally a VICReg variance term on the
pooled predictions. Monitored on held-out pairs: retrieval@1 among 1,000 English titles and embedding spread."""
import argparse, glob, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
from train_student import Block, PAD, UNK
MASK = 4; SPECIAL = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]


class Enc(nn.Module):
    """Same parameter names as Student's encoder side (se, sp, enc, ne)."""
    def __init__(self, nv, d, layers, heads, ff, max_s):
        super().__init__()
        self.se = nn.Embedding(nv, d, padding_idx=PAD); self.sp = nn.Embedding(max_s, d)
        self.enc = nn.ModuleList([Block(d, heads, ff, False) for _ in range(layers)]); self.ne = nn.LayerNorm(d)

    def forward(self, x):
        pos = torch.arange(x.shape[1]).unsqueeze(0); h = self.se(x) + self.sp(pos)
        bias = ((x == PAD).float() * -1e9)[:, None, None, :]
        for b in self.enc:
            h = b(h, bias)
        return self.ne(h)


class Pred(nn.Module):
    def __init__(self, d, layers, heads, ff, max_s):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, ff, False) for _ in range(layers)])
        self.q = nn.Parameter(torch.randn(d) * 0.02); self.g = nn.Parameter(torch.randn(d) * 0.02)
        self.qp = nn.Embedding(max_s, d); self.n = nn.LayerNorm(d)

    def forward(self, ctx, ctx_pad, qpos=None):
        """ctx (B,S,d), ctx_pad (B,S) bool; qpos (B,M) positions to predict (-1 = none). Returns global (B,d), slots (B,M,d)."""
        B = ctx.shape[0]; parts = [self.g.expand(B, 1, -1), ctx]; pads = [torch.zeros(B, 1, dtype=torch.bool), ctx_pad]
        if qpos is not None:
            parts.append(self.q + self.qp(qpos.clamp(min=0))); pads.append(qpos < 0)
        h = torch.cat(parts, 1); bias = (torch.cat(pads, 1).float() * -1e9)[:, None, None, :]
        for b in self.blocks:
            h = b(h, bias)
        h = self.n(h)
        return h[:, 0], (h[:, 1 + ctx.shape[1]:] if qpos is not None else None)


def vicreg(z, cov_w=0.04):
    """Variance (std of every dimension >= 1) + covariance (dimensions decorrelated) terms of VICReg."""
    z = z - z.mean(0); std = torch.sqrt(z.var(0) + 1e-4); var = Fn.relu(1 - std).mean()
    c = (z.T @ z) / max(1, z.shape[0] - 1); off = c - torch.diag(torch.diag(c))
    return var + cov_w * (off ** 2).sum() / z.shape[1]


def pool(states, pad):
    m = (~pad).float().unsqueeze(-1)
    return (states * m).sum(1) / m.sum(1).clamp(min=1)


def load_pairs(tag, split):
    return [json.loads(l) for l in open(os.path.join(W, f"pairs_{tag}_{split}.jsonl"), encoding="utf-8")]


def batches(items, max_tokens, rng, key):
    items = sorted(items, key=key); out, cur, longest = [], [], 0
    for x in items:
        l = key(x)
        if cur and max(longest, l) * (len(cur) + 1) > max_tokens:
            out.append(cur); cur, longest = [], 0
        cur.append(x); longest = max(longest, l)
    if cur:
        out.append(cur)
    rng.shuffle(out); return out


def pad_ids(seqs):
    S = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (S - len(s)) for s in seqs])


def span_mask(n, ratio, rng):
    """Positions to hide: spans of 3-8 characters until about `ratio` of the text is covered."""
    hide = set(); target = max(1, int(n * ratio)); tries = 0
    while len(hide) < target and tries < 50:
        L = rng.randint(3, 8); s = rng.randrange(0, max(1, n - L + 1)); hide.update(range(s, min(n, s + L))); tries += 1
    return sorted(hide)


class JEPA:
    def __init__(self, vocab, d=384, layers=6, heads=6, ff=1536, max_s=96, pred_layers=2):
        self.vocab = vocab; self.vi = {c: i for i, c in enumerate(vocab)}; self.max_s = max_s
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff, max_s=max_s, pred_layers=pred_layers)
        self.E = Enc(len(vocab), d, layers, heads, ff, max_s); self.P = Pred(d, pred_layers, heads, ff, max_s)
        self.T = Enc(len(vocab), d, layers, heads, ff, max_s); self.T.load_state_dict(self.E.state_dict())
        self.center = torch.zeros(d)                   # running mean of pooled targets (DINO-style centering)
        for p in self.T.parameters():
            p.requires_grad_(False)

    def ids(self, s):
        return [self.vi.get(c, UNK) for c in s[: self.max_s]] or [UNK]

    @torch.no_grad()
    def ema(self, m):
        for pt, pe in zip(self.T.parameters(), self.E.parameters()):
            pt.mul_(m).add_(pe.detach(), alpha=1 - m)

    @torch.no_grad()
    def embed_target(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            x = pad_ids([self.ids(t) for t in texts[i:i + bs]]); out.append(Fn.normalize(pool(self.T(x), x == PAD) - self.center, dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def embed_query(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            x = pad_ids([self.ids(t) for t in texts[i:i + bs]]); g, _ = self.P(self.E(x), x == PAD); out.append(Fn.normalize(g, dim=-1))
        return torch.cat(out)

    def save(self, path, extra=None):
        torch.save({"E": self.E.state_dict(), "P": self.P.state_dict(), "T": self.T.state_dict(), "vocab": self.vocab,
                    "cfg": self.cfg, "center": self.center, **(extra or {})}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, map_location="cpu"); c = ck["cfg"]
        j = JEPA(ck["vocab"], c["d"], c["layers"], c["heads"], c["ff"], c["max_s"], c["pred_layers"])
        j.E.load_state_dict(ck["E"]); j.P.load_state_dict(ck["P"]); j.T.load_state_dict(ck["T"])
        if "center" in ck:
            j.center = ck["center"]
        return j, ck


def monitor(j, dev):
    """Held-out: retrieval@1 of foreign -> English among the dev titles, and spread of the target embeddings."""
    j.E.eval(); j.P.eval()
    q = j.embed_query([p["src"] for p in dev]); t = j.embed_target([p["tgt"] for p in dev])
    r1 = (q @ t.T).argmax(1).eq(torch.arange(len(dev))).float().mean().item()
    spread = t.std(0).mean().item() * math.sqrt(t.shape[1])        # ~1 healthy, -> 0 collapsed
    j.E.train(); j.P.train()
    return {"retrieval@1": round(r1, 4), "spread": round(spread, 3)}


def pretrain(a):
    torch.set_num_threads(a.threads); torch.manual_seed(0); rng = random.Random(0)
    out = os.path.join(ROOT, "models", f"jepa_{a.tag}"); os.makedirs(out, exist_ok=True)
    train = [p for p in load_pairs(a.data, "train") if p["src"] and p["tgt"]]
    dev_all = [p for p in load_pairs(a.data, "dev") if p["kind"] in ("bt", "real")]; random.Random(5).shuffle(dev_all); dev = dev_all[:1000]
    ck_path = os.path.join(out, "jepa.pt"); log = {"runs": [], "evals": [], "args": vars(a)}
    if a.resume and os.path.exists(ck_path):
        j, ck = JEPA.load(ck_path); step0 = ck["step"]; done_min = ck["minutes"]; log = ck.get("log", log)
        a.minutes = max(5.0, a.total_min - done_min)              # restart-safe: only the time still owed
        print(f"resume from step {step0}, {done_min:.0f} min done, {a.minutes:.0f} min left", flush=True)
    else:
        from collections import Counter
        cnt = Counter(c for p in train for s in (p["src"], p["tgt"]) for c in s)
        vocab = SPECIAL + [c for c, n in cnt.most_common() if n >= 3]
        j = JEPA(vocab, a.d, a.layers, a.heads, a.ff, 96, a.pred_layers); step0 = 0; done_min = 0.0
    params = list(j.E.parameters()) + list(j.P.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.04)
    if a.resume and os.path.exists(ck_path) and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    print(f"JEPA {a.tag}: encoder {sum(p.numel() for p in j.E.parameters()) / 1e6:.1f}M, predictor "
          f"{sum(p.numel() for p in j.P.parameters()) / 1e6:.1f}M, vocab {len(j.vocab)}, pairs {len(train)}", flush=True)
    texts = [p["tgt"] for p in train] + [p["src"] for p in train]
    t0 = time.time(); t_end = t0 + a.minutes * 60; step = step0; next_ck = t0 + a.ckpt_min * 60; L = []
    while time.time() < t_end:
        PB = batches(train, a.tokens, rng, key=lambda p: max(len(p["src"]), len(p["tgt"])))
        TB = batches(texts, a.tokens, rng, key=len)
        for pb, tb in zip(PB, TB):
            now = time.time()
            if now >= t_end:
                break
            frac = min(1.0, (done_min + (now - t0) / 60) / a.total_min)
            lr = a.lr * min(1, (step + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            m = 1 - (1 - a.ema) * (0.5 * (1 + math.cos(math.pi * frac)))          # EMA momentum a.ema -> 1
            # cross-lingual: foreign label -> pooled English target embedding
            xs = pad_ids([j.ids(p["src"]) for p in pb]); xt = pad_ids([j.ids(p["tgt"]) for p in pb])
            with torch.no_grad():
                zr = pool(j.T(xt), xt == PAD); j.center = 0.9 * j.center + 0.1 * zr.mean(0)
                zt = Fn.normalize(zr - j.center, dim=-1)
            hs = j.E(xs); g, _ = j.P(hs, xs == PAD); gp = Fn.normalize(g, dim=-1)
            l_cross = (2 - 2 * (gp * zt).sum(-1)).mean()
            # anti-collapse on what gradients reach: the predictor output and the context encoder's pooled states
            l_var = (vicreg(g) + vicreg(pool(hs, xs == PAD))) if a.vic > 0 and len(pb) > 2 else torch.zeros(())
            # masked spans: hidden character spans -> target encoder states at those positions
            seqs = [j.ids(t) for t in tb]; hides = [span_mask(len(s), a.mask, rng) for s in seqs]
            ctx = []
            for s_, h in zip(seqs, hides):
                hs = set(h); ctx.append([MASK if i in hs else c for i, c in enumerate(s_)])
            xc = pad_ids(ctx); xf = pad_ids(seqs); M = max(len(h) for h in hides)
            qpos = torch.tensor([h + [-1] * (M - len(h)) for h in hides])
            with torch.no_grad():
                tf = Fn.layer_norm(j.T(xf), (j.cfg["d"],))
                tgt = torch.stack([tf[b, qpos[b].clamp(min=0)] for b in range(len(seqs))])
            _, sl = j.P(j.E(xc), xc == PAD, qpos)
            valid = (qpos >= 0).float().unsqueeze(-1)
            l_mask = (Fn.smooth_l1_loss(sl, tgt, reduction="none") * valid).sum() / valid.sum().clamp(min=1) / j.cfg["d"] * 10
            loss = l_cross + a.w_mask * l_mask + a.vic * l_var
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); j.ema(m); step += 1
            L.append((l_cross.item(), l_mask.item(), float(l_var)))
            if step % 100 == 0:
                c = [sum(x[i] for x in L[-100:]) / len(L[-100:]) for i in range(3)]
                print(f"step {step} cross {c[0]:.3f} mask {c[1]:.3f} var {c[2]:.3f} lr {lr:.2e} ema {m:.4f} "
                      f"{(t_end - now) / 60:.0f} min left", flush=True)
            if now >= next_ck:
                r = monitor(j, dev); r["step"] = step; log["evals"].append(r); print("dev", r, flush=True)
                mins = done_min + (time.time() - t0) / 60
                j.save(ck_path + ".tmp", {"step": step, "minutes": mins, "opt": opt.state_dict(), "log": log})
                os.replace(ck_path + ".tmp", ck_path); next_ck = time.time() + a.ckpt_min * 60
    r = monitor(j, dev); r["step"] = step; log["evals"].append(r); print("dev (end)", r, flush=True)
    log["runs"].append({"minutes": a.minutes, "to_step": step})
    j.save(ck_path, {"step": step, "minutes": done_min + a.minutes, "opt": opt.state_dict(), "log": log})
    json.dump(log, open(os.path.join(out, "jepa_log.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["pretrain"])
    ap.add_argument("--data", default="MAX"); ap.add_argument("--tag", default="main")
    ap.add_argument("--minutes", type=float, default=180); ap.add_argument("--total-min", type=float, default=None)
    ap.add_argument("--d", type=int, default=384); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6); ap.add_argument("--ff", type=int, default=1536)
    ap.add_argument("--pred-layers", type=int, default=2); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tokens", type=int, default=4000); ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--mask", type=float, default=0.3); ap.add_argument("--w-mask", type=float, default=1.0)
    ap.add_argument("--vic", type=float, default=1.0); ap.add_argument("--ckpt-min", type=float, default=15)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); a.total_min = a.total_min or a.minutes
    pretrain(a)


if __name__ == "__main__":
    main()
