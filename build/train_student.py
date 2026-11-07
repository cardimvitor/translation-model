"""Train the character-level student (foreign -> English) on CPU, export ONNX (fp32 + int8).

Model: Transformer, d=256, 4 encoder / 3 decoder layers, 4 heads, ff 1024, learned positions.
Source vocab: characters seen in training (rare ones -> <unk>); target vocab: English characters.
Usage: python train_student.py [minutes]
"""
import json, math, os, random, sys, time
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
W = os.path.join(HERE, "work"); OUT = os.path.join(ROOT, "models", "student"); os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); random.seed(0)
MAX_S, MAX_T = 64, 64
PAD, UNK, BOS, EOS = 0, 1, 2, 3


def load(split):
    return [json.loads(l) for l in open(os.path.join(W, f"pairs_{split}.jsonl"), encoding="utf-8")]


class MHA(nn.Module):
    """Plain multi-head attention with shape-agnostic reshapes (exports to ONNX with dynamic lengths)."""
    def __init__(self, d, h):
        super().__init__(); self.h = h; self.hd = d // h
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, x, kv, bias):
        B = x.shape[0]
        q = self.q(x).view(B, -1, self.h, self.hd).transpose(1, 2)
        k = self.k(kv).view(B, -1, self.h, self.hd).transpose(1, 2)
        v = self.v(kv).view(B, -1, self.h, self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd) + bias, dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, -1, self.h * self.hd))


class Block(nn.Module):
    def __init__(self, d, h, ff, cross):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.sa = MHA(d, h)
        self.cross = cross
        if cross:
            self.n2 = nn.LayerNorm(d); self.ca = MHA(d, h)
        self.n3 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(0.1)

    def forward(self, x, self_bias, mem=None, mem_bias=None):
        y = self.n1(x); x = x + self.drop(self.sa(y, y, self_bias))
        if self.cross:
            x = x + self.drop(self.ca(self.n2(x), mem, mem_bias))
        return x + self.drop(self.ff(self.n3(x)))


class Student(nn.Module):
    def __init__(self, ns, nt, d=256, enc=4, dec=3, heads=4, ff=1024):
        super().__init__()
        self.se = nn.Embedding(ns, d, padding_idx=PAD); self.te = nn.Embedding(nt, d, padding_idx=PAD)
        self.sp = nn.Embedding(MAX_S, d); self.tp = nn.Embedding(MAX_T, d)
        self.enc = nn.ModuleList([Block(d, heads, ff, False) for _ in range(enc)])
        self.dec = nn.ModuleList([Block(d, heads, ff, True) for _ in range(dec)])
        self.ne = nn.LayerNorm(d); self.nd = nn.LayerNorm(d); self.out = nn.Linear(d, nt)

    @staticmethod
    def pad_bias(src):
        return ((src == PAD).float() * -1e9)[:, None, None, :]          # (B,1,1,S)

    def encode(self, src):
        pos = torch.arange(src.shape[1]).unsqueeze(0)
        x = self.se(src) + self.sp(pos); bias = self.pad_bias(src)
        for b in self.enc:
            x = b(x, bias)
        return self.ne(x)

    def decode(self, tgt, mem, src):
        T = tgt.shape[1]; pos = torch.arange(T).unsqueeze(0)
        causal = torch.triu(torch.ones(T, T), 1) * -1e9
        x = self.te(tgt) + self.tp(pos); mb = self.pad_bias(src)
        for b in self.dec:
            x = b(x, causal[None, None], mem, mb)
        return self.out(self.nd(x))

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src)


def main(minutes):
    train, dev = load("train"), load("dev")
    from collections import Counter
    cs = Counter(c for p in train for c in p["src"]); ct = Counter(c for p in train for c in p["tgt"])
    svocab = ["<pad>", "<unk>", "<s>", "</s>"] + [c for c, n in cs.most_common() if n >= 2]
    tvocab = ["<pad>", "<unk>", "<s>", "</s>"] + [c for c, n in ct.most_common() if n >= 2]
    si = {c: i for i, c in enumerate(svocab)}; ti = {c: i for i, c in enumerate(tvocab)}
    enc_s = lambda s: [si.get(c, UNK) for c in s[: MAX_S]]
    enc_t = lambda s: [BOS] + [ti.get(c, UNK) for c in s[: MAX_T - 2]] + [EOS]
    data = [(enc_s(p["src"]), enc_t(p["tgt"])) for p in train if p["src"] and p["tgt"]]
    # weight product pairs up (they are the student's job), places down
    weights = {"bt": 3, "gloss": 2, "gaz": 1}
    data = [x for p, x in zip(train, data) for _ in range(weights[p["kind"]])]
    model = Student(len(svocab), len(tvocab))
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | src vocab {len(svocab)} tgt vocab {len(tvocab)} | pairs {len(data)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    torch.set_num_threads(4)
    t_end = time.time() + minutes * 60; step = 0; B = 96
    data.sort(key=lambda x: len(x[0]))
    batches = [data[i:i + B] for i in range(0, len(data), B)]
    total_steps_guess = None
    while time.time() < t_end:
        random.shuffle(batches)
        for b in batches:
            if time.time() >= t_end:
                break
            S = max(len(x[0]) for x in b); T = max(len(x[1]) for x in b)
            src = torch.tensor([x[0] + [PAD] * (S - len(x[0])) for x in b])
            tgt = torch.tensor([x[1] + [PAD] * (T - len(x[1])) for x in b])
            frac = 1 - (t_end - time.time()) / (minutes * 60)
            lr = 1e-3 * min(1, (step + 1) / 400) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            model.train()
            logits = model(src, tgt[:, :-1])
            loss = lossf(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            if step % 200 == 0:
                print(f"step {step} loss {loss.item():.3f} lr {lr:.2e} {(1 - frac) * minutes:.0f} min left", flush=True)
    model.eval()
    torch.save({"model": model.state_dict(), "svocab": svocab, "tvocab": tvocab}, os.path.join(OUT, "student.pt"))
    json.dump({"src": svocab, "tgt": tvocab, "max_src": MAX_S, "max_tgt": MAX_T, "pad": PAD, "unk": UNK, "bos": BOS, "eos": EOS},
              open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    export(model, len(svocab))
    print("saved", flush=True)


class Enc(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, src): return self.m.encode(src)


class Dec(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, tgt, mem, src): return self.m.decode(tgt, mem, src)[:, -1, :]


def export(model, ns):
    src = torch.tensor([[5, 6, 7, 8, 9, 10, 11]]); tgt = torch.tensor([[BOS, 5, 6]])
    mem = model.encode(src)
    torch.onnx.export(Enc(model), (src,), os.path.join(OUT, "encoder.onnx"), input_names=["src"], output_names=["mem"],
                      dynamic_axes={"src": {1: "S"}, "mem": {1: "S"}}, opset_version=17, dynamo=False)
    torch.onnx.export(Dec(model), (tgt, mem, src), os.path.join(OUT, "decoder.onnx"), input_names=["tgt", "mem", "src"],
                      output_names=["logits"], dynamic_axes={"tgt": {1: "T"}, "mem": {1: "S"}, "src": {1: "S"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for n in ("encoder", "decoder"):
        quantize_dynamic(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_int8.onnx"), weight_type=QuantType.QInt8)
    for f in sorted(os.listdir(OUT)):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 2), "MB")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60)
