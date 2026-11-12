"""Export the fine-tuned translator (build/ft_marian.py) for the phone runtime, in the same shape as the students:
  encoder: src -> ck, cv (cross-attention keys/values per decoder layer), src_bias
  decoder: one step with a self-attention cache: tok, pos, ck, cv, src_bias, pk, pv -> logits, nk, nv
then int8 quantization, vocab.json ("spm": true) and spm_src.json (source SentencePiece pieces + scores).

  python build/export_marian.py --ckpt models/marian_ft/latest.pt --out student_V6 [--avg 3] [--fp32]
--avg N averages the last N step_*.pt checkpoints instead of --ckpt."""
import argparse, glob, json, math, os, shutil, sys
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_marian as FT
from ft_marian import EOS, UNK, PAD


def heads(x, h):
    b, t, d = x.shape
    return x.view(b, t, h, d // h).transpose(1, 2)


class EncKV(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, src):
        mm = self.m.model; e = mm.encoder(input_ids=src).last_hidden_state
        h = mm.decoder.layers[0].encoder_attn.num_heads
        ck = torch.stack([heads(l.encoder_attn.k_proj(e), h) for l in mm.decoder.layers])
        cv = torch.stack([heads(l.encoder_attn.v_proj(e), h) for l in mm.decoder.layers])
        return ck, cv, torch.zeros(1, 1, 1, src.shape[1])


class DecStep(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m

    def forward(self, tok, pos, ck, cv, src_bias, pk, pv):
        mm = self.m.model; d = mm.decoder; h = d.layers[0].self_attn.num_heads; hd = d.layers[0].self_attn.head_dim
        x = d.embed_tokens(tok) * d.embed_scale + d.embed_positions.weight[pos]
        nk, nv = [], []
        for l, b in enumerate(d.layers):
            sa = b.self_attn
            q = heads(sa.q_proj(x), h); k = torch.cat([pk[l], heads(sa.k_proj(x), h)], 2); v = torch.cat([pv[l], heads(sa.v_proj(x), h)], 2)
            nk.append(k); nv.append(v)
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
            x = b.self_attn_layer_norm(x + sa.out_proj((a @ v).transpose(1, 2).reshape(x.shape[0], -1, h * hd)))
            ca = b.encoder_attn; q = heads(ca.q_proj(x), h)
            a = torch.softmax(q @ ck[l].transpose(-1, -2) / math.sqrt(hd) + src_bias, dim=-1)
            x = b.encoder_attn_layer_norm(x + ca.out_proj((a @ cv[l]).transpose(1, 2).reshape(x.shape[0], -1, h * hd)))
            x = b.final_layer_norm(x + b.fc2(b.activation_fn(b.fc1(x))))
        return (self.m.lm_head(x) + self.m.final_logits_bias)[:, -1, :], torch.stack(nk), torch.stack(nv)


def greedy(enc, dec, ids, L, h, hd, max_new=48):
    """Reference loop on the torch wrappers (same as the runtimes)."""
    with torch.no_grad():
        ck, cv, sb = enc(torch.tensor([ids])); pk = pv = torch.zeros(L, 1, h, 0, hd); tok = PAD; out = []
        for t in range(max_new):
            lg, pk, pv = dec(torch.tensor([[tok]]), torch.tensor([[t]]), ck, cv, sb, pk, pv)
            lg[0, PAD] = -1e9; lg[0, UNK] = -1e9; tok = int(lg[0].argmax())
            if tok == EOS:
                break
            out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="marian_ft"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--avg", type=int, default=0); ap.add_argument("--out", default="student_V6"); ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args(); d = FT.out_dir(a.tag); OUT = os.path.join(ROOT, "models", a.out); os.makedirs(OUT, exist_ok=True)
    tok, m, _ = FT.load(d, a.ckpt or os.path.join(d, "latest.pt"))
    if a.avg:
        cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))[-a.avg:]; sds = [torch.load(c, map_location="cpu")["model"] for c in cks]
        m.load_state_dict({k: sum(s[k].float() for s in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]})
        print("averaged", [os.path.basename(c) for c in cks])
    m.eval(); enc, dec = EncKV(m).eval(), DecStep(m).eval()
    L = len(m.model.decoder.layers); h = m.model.decoder.layers[0].self_attn.num_heads; hd = m.model.decoder.layers[0].self_attn.head_dim
    # the step-by-step wrappers must give the same tokens as the model's own generate()
    tests = ["kabellose kopfhörer", "сумка кожаная женская", "電気ケトル", "chemise en coton pour homme"]
    ref = FT.translate(tok, m, tests); mine = [tok.dec(greedy(enc, dec, tok.enc([t])[0], L, h, hd)) for t in tests]
    print("generate():", ref); print("step loop: ", mine); assert ref == mine, "wrapper mismatch"
    src = torch.tensor([[5, 6, 7, 8, 9, 10, EOS]])
    with torch.no_grad():
        ck, cv, sb = enc(src)
    torch.onnx.export(enc, (src,), os.path.join(OUT, "encoder.onnx"), input_names=["src"], output_names=["ck", "cv", "src_bias"],
                      dynamic_axes={"src": {1: "S"}, "ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}}, opset_version=17, dynamo=False)
    t = torch.tensor([[PAD]]); pos = torch.tensor([[2]]); pk = torch.zeros(L, 1, h, 2, hd); pv = torch.zeros(L, 1, h, 2, hd)
    torch.onnx.export(dec, (t, pos, ck, cv, sb, pk, pv), os.path.join(OUT, "decoder.onnx"),
                      input_names=["tok", "pos", "ck", "cv", "src_bias", "pk", "pv"], output_names=["logits", "nk", "nv"],
                      dynamic_axes={"ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}, "pk": {3: "T"}, "pv": {3: "T"}, "nk": {3: "T1"}, "nv": {3: "T1"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for n in ("encoder", "decoder"):
        quantize_dynamic(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_int8.onnx"), weight_type=QuantType.QInt8)
        os.replace(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_fp32.onnx.bak"))
    json.dump({"spm": True, "src": tok.vocab, "tgt": tok.vocab, "max_src": 64, "max_tgt": 49, "pad": PAD, "unk": UNK, "bos": PAD, "eos": EOS,
               "kv": True, "layers": L, "heads": h, "head_dim": hd}, open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    from sentencepiece import sentencepiece_model_pb2 as pb
    mp = pb.ModelProto(); mp.ParseFromString(open(os.path.join(d, "source.spm"), "rb").read()); P = [p for p in mp.pieces if p.type == 1]
    json.dump({"pieces": [p.piece for p in P], "scores": [p.score for p in P]}, open(os.path.join(OUT, "spm_src.json"), "w", encoding="utf-8"), ensure_ascii=False)
    torch.save({"model": m.state_dict()}, os.path.join(OUT, "student.pt"))
    for f in ("source.spm", "target.spm"):
        shutil.copy(os.path.join(d, f), os.path.join(OUT, f))
    for f in sorted(os.listdir(OUT)):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
