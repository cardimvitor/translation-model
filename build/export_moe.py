"""Export v7 (v6 + language experts) for the phone runtime. Same graphs as v6 plus four inputs that carry the chosen
expert: ed, eu (encoder layers: d x r, r x d) and dd, du (decoder layers). Zeros = plain v6. Each language's expert
is one small file, models/<out>/experts/<lang>.bin (float32: ed, eu, dd, du), loaded only when that language is seen.
  python build/export_moe.py --out student_V7"""
import argparse, json, math, os, shutil, sys
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_marian as FT, ft_moe as MOE
from ft_marian import EOS, UNK, PAD
from export_marian import heads


def adapt(h, down, up):
    return h + torch.relu(h @ down) @ up


class EncKV(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m; self.cur = None
        for k, layer in enumerate(m.model.encoder.layers):
            layer.register_forward_hook(lambda mod, inp, out, k=k: ((adapt(out[0], self.cur[0][k], self.cur[1][k]),) + tuple(out[1:])) if isinstance(out, tuple) else adapt(out, self.cur[0][k], self.cur[1][k]))

    def forward(self, src, ed, eu):
        self.cur = (ed, eu); mm = self.m.model; e = mm.encoder(input_ids=src).last_hidden_state
        h = mm.decoder.layers[0].encoder_attn.num_heads
        ck = torch.stack([heads(l.encoder_attn.k_proj(e), h) for l in mm.decoder.layers])
        cv = torch.stack([heads(l.encoder_attn.v_proj(e), h) for l in mm.decoder.layers])
        return ck, cv, torch.zeros(1, 1, 1, src.shape[1])


class DecStep(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, tok, pos, ck, cv, src_bias, pk, pv, dd, du):
        mm = self.m.model; d = mm.decoder; h = d.layers[0].self_attn.num_heads; hd = d.layers[0].self_attn.head_dim
        x = d.embed_tokens(tok) * d.embed_scale + d.embed_positions.weight[pos]; nk, nv = [], []
        for l, b in enumerate(d.layers):
            sa = b.self_attn
            q = heads(sa.q_proj(x), h); k = torch.cat([pk[l], heads(sa.k_proj(x), h)], 2); v = torch.cat([pv[l], heads(sa.v_proj(x), h)], 2)
            nk.append(k); nv.append(v)
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
            x = b.self_attn_layer_norm(x + sa.out_proj((a @ v).transpose(1, 2).reshape(x.shape[0], -1, h * hd)))
            ca = b.encoder_attn; q = heads(ca.q_proj(x), h)
            a = torch.softmax(q @ ck[l].transpose(-1, -2) / math.sqrt(hd) + src_bias, dim=-1)
            x = b.encoder_attn_layer_norm(x + ca.out_proj((a @ cv[l]).transpose(1, 2).reshape(x.shape[0], -1, h * hd)))
            x = adapt(b.final_layer_norm(x + b.fc2(b.activation_fn(b.fc1(x)))), dd[l], du[l])
        return (self.m.lm_head(x) + self.m.final_logits_bias)[:, -1, :], torch.stack(nk), torch.stack(nv)


def greedy(enc, dec, ids, E, L, h, hd, max_new=48):
    with torch.no_grad():
        ck, cv, sb = enc(torch.tensor([ids]), E[0], E[1]); pk = pv = torch.zeros(L, 1, h, 0, hd); tok = PAD; out = []
        for t in range(max_new):
            lg, pk, pv = dec(torch.tensor([[tok]]), torch.tensor([[t]]), ck, cv, sb, pk, pv, E[2], E[3])
            lg[0, PAD] = -1e9; lg[0, UNK] = -1e9; tok = int(lg[0].argmax())
            if tok == EOS:
                break
            out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="moe_v7"); ap.add_argument("--out", default="student_V7"); ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args(); OUT = os.path.join(ROOT, "models", a.out); os.makedirs(os.path.join(OUT, "experts"), exist_ok=True)
    tok, mref, ex, _ = MOE.load(a, os.path.join(FT.out_dir(a.tag), "latest.pt"))                     # reference: hooks + routing
    _, m, _ = FT.load(FT.out_dir("marian_ft"), os.path.join(ROOT, "models", "student_V6", "student.pt")); m.eval()
    ne = len(m.model.encoder.layers); L = len(m.model.decoder.layers); h = m.model.decoder.layers[0].self_attn.num_heads; hd = m.model.decoder.layers[0].self_attn.head_dim
    enc, dec = EncKV(m).eval(), DecStep(m).eval()
    def expert(i):
        return (ex.down[i, :ne].detach(), ex.up[i, :ne].detach(), ex.down[i, ne:].detach(), ex.up[i, ne:].detach()) if i else \
            tuple(torch.zeros_like(t) for t in (ex.down[0, :ne], ex.up[0, :ne], ex.down[0, ne:], ex.up[0, ne:]))
    tests = [("kabellose kopfhörer", "de"), ("сумка кожаная женская", "ru"), ("電気ケトル", "ja"),
             ("무선 마우스", "ko"), ("chemise en coton pour homme", "fr"), ("yoga mat", "none")]
    ref = MOE.translate(tok, mref, ex, [t for t, _ in tests], [MOE.LANGS.index(l) for _, l in tests])
    mine = [tok.dec(greedy(enc, dec, tok.enc([t])[0], expert(MOE.LANGS.index(l)), L, h, hd)) for t, l in tests]
    print("generate():", ref); print("step loop: ", mine); assert ref == mine, "wrapper mismatch"
    src = torch.tensor([[5, 6, 7, 8, 9, 10, EOS]]); E = expert(1)
    with torch.no_grad():
        ck, cv, sb = enc(src, E[0], E[1])
    torch.onnx.export(enc, (src, E[0], E[1]), os.path.join(OUT, "encoder.onnx"), input_names=["src", "ed", "eu"], output_names=["ck", "cv", "src_bias"],
                      dynamic_axes={"src": {1: "S"}, "ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}}, opset_version=17, dynamo=False)
    t = torch.tensor([[PAD]]); pos = torch.tensor([[2]]); pk = torch.zeros(L, 1, h, 2, hd); pv = torch.zeros(L, 1, h, 2, hd)
    torch.onnx.export(dec, (t, pos, ck, cv, sb, pk, pv, E[2], E[3]), os.path.join(OUT, "decoder.onnx"),
                      input_names=["tok", "pos", "ck", "cv", "src_bias", "pk", "pv", "dd", "du"], output_names=["logits", "nk", "nv"],
                      dynamic_axes={"ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}, "pk": {3: "T"}, "pv": {3: "T"}, "nk": {3: "T1"}, "nv": {3: "T1"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for n in ("encoder", "decoder"):
        quantize_dynamic(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_int8.onnx"), weight_type=QuantType.QInt8, per_channel=True)
        os.remove(os.path.join(OUT, f"{n}.onnx"))
    for i, lang in enumerate(MOE.LANGS):
        if i:
            np.concatenate([t.numpy().astype(np.float32).ravel() for t in expert(i)]).tofile(os.path.join(OUT, "experts", f"{lang}.bin"))
    v6 = json.load(open(os.path.join(ROOT, "models", "student_V6", "vocab.json"), encoding="utf-8"))
    v6["experts"] = {"langs": MOE.LANGS[1:], "rank": a.rank, "d": m.config.d_model, "enc_layers": ne, "dec_layers": L, "min_conf": MOE.MIN_CONF}
    json.dump(v6, open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    shutil.copy(os.path.join(ROOT, "models", "student_V6", "spm_src.json"), OUT)
    torch.save({"experts": ex.state_dict()}, os.path.join(OUT, "experts.pt"))
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f); print(f, round(os.path.getsize(p) / 1e6, 2), "MB") if os.path.isfile(p) else print(f, len(os.listdir(p)), "files,", round(os.path.getsize(os.path.join(p, "ko.bin")) / 1e6, 2), "MB each")


if __name__ == "__main__":
    main()
