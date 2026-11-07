"""Batched greedy NLLB-600M translation with onnxruntime (build time, CPU)."""
import json, os, sys, time, math
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
M = os.path.join(ROOT, "models", "nllb-600m")
CODES = {"zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "ru": "rus_Cyrl", "hi": "hin_Deva",
         "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "it": "ita_Latn",
         "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "pt": "por_Latn", "en": "eng_Latn"}


class NLLB:
    def __init__(self, threads=4):
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        self.enc = ort.InferenceSession(os.path.join(M, "onnx", "encoder_model_quantized.onnx"), o, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(M, "onnx", "decoder_model_merged_quantized.onnx"), o, providers=["CPUExecutionProvider"])
        tj = json.load(open(os.path.join(M, "tokenizer.json"), encoding="utf-8"))
        self.tok = Tokenizer.from_str(json.dumps(tj))
        self.vocab = {t["content"]: t["id"] for t in tj["added_tokens"]}
        cfg = json.load(open(os.path.join(M, "config.json")))
        self.eos, self.pad, self.start = cfg["eos_token_id"], cfg["pad_token_id"], cfg["decoder_start_token_id"]
        self.L, self.H, self.hd = cfg["decoder_layers"], cfg["decoder_attention_heads"], cfg["d_model"] // cfg["decoder_attention_heads"]
        self.out_names = [o.name for o in self.dec.get_outputs()]

    def translate(self, texts, src, tgt, max_new=40):
        B = len(texts)
        ids = [[self.vocab[CODES[src]]] + self.tok.encode(t, add_special_tokens=False).ids[:60] + [self.eos] for t in texts]
        T = max(map(len, ids))
        inp = np.full((B, T), self.pad, np.int64); mask = np.zeros((B, T), np.int64)
        for i, x in enumerate(ids):
            inp[i, :len(x)] = x; mask[i, :len(x)] = 1
        hs = self.enc.run(None, {"input_ids": inp, "attention_mask": mask})[0]
        past = {f"past_key_values.{l}.{a}.{kv}": np.zeros((B, self.H, 0, self.hd), np.float32)
                for l in range(self.L) for a in ("decoder", "encoder") for kv in ("key", "value")}
        cur = np.full((B, 1), self.start, np.int64)
        outs = [[] for _ in range(B)]; done = np.zeros(B, bool); lp_sum = np.zeros(B); n_tok = np.zeros(B)
        forced = self.vocab[CODES[tgt]]
        for step in range(max_new + 1):
            feed = {"encoder_attention_mask": mask, "input_ids": cur, "encoder_hidden_states": hs, "use_cache_branch": np.array([step > 0])}
            feed.update(past)
            res = self.dec.run(None, feed)
            logits = res[0][:, -1].astype(np.float32)
            for name, val in zip(self.out_names[1:], res[1:]):
                key = name.replace("present.", "past_key_values.")
                if step == 0 or ".decoder." in key:
                    past[key] = val
            if step == 0:
                nxt = np.full(B, forced, np.int64)
            else:
                logits[:, self.pad] = -1e9
                m = logits.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
                nxt = logits.argmax(1)
                for i in range(B):
                    if not done[i]:
                        if nxt[i] == self.eos:
                            done[i] = True
                        else:
                            outs[i].append(int(nxt[i])); lp_sum[i] += logits[i, nxt[i]] - lse[i]; n_tok[i] += 1
                nxt = np.where(done, self.pad, nxt)
            if done.all():
                break
            cur = nxt.reshape(B, 1).astype(np.int64)
        return [(self.tok.decode(o, skip_special_tokens=True).strip(), float(math.exp(lp_sum[i] / max(1, n_tok[i]))))
                for i, o in enumerate(outs)]


if __name__ == "__main__":
    n = NLLB()
    texts = ["women's black leather bag", "wireless headphones", "men's dark blue tie", "children's light beige trousers",
             "Love You Mom Foil Balloon", "stainless steel water bottle", "Local Boy Toddler Volley Shorts in Black", "phone case"] * 4
    for tgt in ("es", "zh", "ru"):
        t0 = time.time(); r = n.translate(texts, "en", tgt); dt = time.time() - t0
        print(tgt, f"{dt / len(texts) * 1000:.0f} ms/item (batch {len(texts)})", r[:4])
