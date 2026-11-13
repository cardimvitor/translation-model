"""Export v8 for the phone runtime: encoder (with the chosen language expert as inputs ed, eu) and a one-step
decoder with a self-attention cache; per-channel int8; one small expert file per language (encoder only).
  python build/export_v8.py --out student_V8 [--ckpt models/v8/latest.pt]"""
import argparse, json, os, shutil, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_marian as FT, ft_moe as MOE, ft_v8 as V8
from ft_marian import EOS, UNK, PAD
from export_marian import DecStep
from export_moe import EncKV


def greedy(enc, dec, ids, E, L, h, hd, max_new=48):
    with torch.no_grad():
        ck, cv, sb = enc(torch.tensor([ids]), E[0], E[1]); pk = pv = torch.zeros(L, 1, h, 0, hd); tok = PAD; out = []
        for t in range(max_new):
            lg, pk, pv = dec(torch.tensor([[tok]]), torch.tensor([[t]]), ck, cv, sb, pk, pv)
            lg[0, PAD] = -1e9; lg[0, UNK] = -1e9; tok = int(lg[0].argmax())
            if tok == EOS:
                break
            out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v8"); ap.add_argument("--out", default="student_V8"); ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(); OUT = os.path.join(ROOT, "models", a.out); os.makedirs(OUT, exist_ok=True)
    ck_path = a.ckpt or os.path.join(V8.out_dir(a.tag), "latest.pt")
    tok, mref, ex, sd = V8.load(a, ck_path)                                        # reference: hooks + routing
    base = torch.load(os.path.join(V8.out_dir(a.tag), "base.pt"), map_location="cpu")
    m = V8.build(len(tok.src), len(tok.tgt), len(base["dec_keep"]), base.get("enc_layers", 6)); m.load_state_dict(sd["model"]); m.eval()
    lite = bool(base.get("lite"))
    ne = len(m.model.encoder.layers); L = len(m.model.decoder.layers); h = m.model.decoder.layers[0].self_attn.num_heads; hd = m.model.decoder.layers[0].self_attn.head_dim
    if lite:                                                # fast model: no experts, plain encoder graph (input: src only)
        from export_marian import EncKV as PlainEnc
        pe = PlainEnc(m).eval(); enc = lambda src, ed=None, eu=None: pe(src)
    else:
        enc = EncKV(m).eval()
    dec = DecStep(m).eval()
    expert = lambda i: (ex.down[i].detach(), ex.up[i].detach()) if i else (torch.zeros_like(ex.down[0]), torch.zeros_like(ex.up[0]))
    tests = [("kabellose kopfhörer", "de"), ("сумка кожаная женская", "ru"), ("電気ケトル", "ja"),
             ("무선 마우스", "ko"), ("chemise en coton pour homme", "fr"), ("yoga mat", "none")]
    ref = V8.translate(tok, mref, ex, [t for t, _ in tests], [MOE.LANGS.index(l) for _, l in tests])
    mine = [tok.dec(greedy(enc, dec, tok.enc([t])[0], expert(MOE.LANGS.index(l)), L, h, hd)) for t, l in tests]
    print("generate():", ref); print("step loop: ", mine); assert ref == mine, "wrapper mismatch"
    src = torch.tensor([[5, 6, 7, 8, 9, 10, EOS]]); E = expert(1)
    with torch.no_grad():
        ck, cv, sb = enc(src, E[0], E[1])
    torch.onnx.export(pe if lite else enc, (src,) if lite else (src, E[0], E[1]), os.path.join(OUT, "encoder.onnx"), input_names=["src"] if lite else ["src", "ed", "eu"], output_names=["ck", "cv", "src_bias"],
                      dynamic_axes={"src": {1: "S"}, "ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}}, opset_version=17, dynamo=False)
    t = torch.tensor([[PAD]]); pos = torch.tensor([[2]]); pk = torch.zeros(L, 1, h, 2, hd); pv = torch.zeros(L, 1, h, 2, hd)
    torch.onnx.export(dec, (t, pos, ck, cv, sb, pk, pv), os.path.join(OUT, "decoder.onnx"),
                      input_names=["tok", "pos", "ck", "cv", "src_bias", "pk", "pv"], output_names=["logits", "nk", "nv"],
                      dynamic_axes={"ck": {3: "S"}, "cv": {3: "S"}, "src_bias": {3: "S"}, "pk": {3: "T"}, "pv": {3: "T"}, "nk": {3: "T1"}, "nv": {3: "T1"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for n in ("encoder", "decoder"):
        quantize_dynamic(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_int8.onnx"), weight_type=QuantType.QInt8, per_channel=True)
        os.replace(os.path.join(OUT, f"{n}.onnx"), os.path.join(OUT, f"{n}_fp32.onnx.bak"))
    if not lite:
        os.makedirs(os.path.join(OUT, "experts"), exist_ok=True)
    for i, lang in enumerate(MOE.LANGS):
        if i and not lite:
            np.concatenate([x.numpy().astype(np.float32).ravel() for x in expert(i)]).tofile(os.path.join(OUT, "experts", f"{lang}.bin"))
    json.dump({"spm": True, "src": tok.src, "tgt": tok.tgt, "max_src": 64, "max_tgt": 49, "pad": PAD, "unk": UNK, "bos": PAD, "eos": EOS,
               "kv": True, "layers": L, "heads": h, "head_dim": hd,
               **({} if lite else {"experts": {"langs": MOE.LANGS[1:], "rank": a.rank, "d": m.config.d_model, "enc_layers": ne, "dec_layers": 0, "min_conf": MOE.MIN_CONF}})},
              open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    shutil.copy(os.path.join(ROOT, "models", "student_V6", "spm_src.json"), OUT)
    torch.save({"model": m.state_dict(), "experts": ex.state_dict(), "dec_keep": base["dec_keep"]}, os.path.join(OUT, "student.pt"))
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f, round(os.path.getsize(p) / 1e6, 2), "MB") if os.path.isfile(p) else print(f, len(os.listdir(p)), "files,", round(os.path.getsize(os.path.join(p, "ko.bin")) / 1e6, 2) if os.path.exists(os.path.join(p, "ko.bin")) else 0, "MB each")


if __name__ == "__main__":
    main()
