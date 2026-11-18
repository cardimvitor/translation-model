"""Tencent Hy-MT2 (Apache-2.0) as a teacher, through llama.cpp (GGUF, CPU).
  from hy_gen import Hy; h = Hy("models/hy-mt2/Hy-MT2-1.8B-Q4_K_M.gguf"); h.translate(["Москва"], "ru", "en")
Greedy decoding with the model card's prompt; returns [(text, 0.5)] (no confidence)."""
import math, os
from llama_cpp import Llama
NAMES = {"zh": "Chinese", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic", "ru": "Russian", "ja": "Japanese",
         "ko": "Korean", "de": "German", "fr": "French", "hi": "Hindi", "tr": "Turkish", "id": "Indonesian",
         "vi": "Vietnamese", "it": "Italian", "pl": "Polish", "th": "Thai", "en": "English"}


class Hy:
    def __init__(self, path, threads=int(os.environ.get("THREADS", 4)), ctx=512):
        self.m = Llama(model_path=path, n_ctx=ctx, n_threads=threads, n_batch=256, logits_all=False, verbose=False)

    def prompt(self, text, tgt):
        if tgt == "zh":
            return f"将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        return (f"Translate the following text into {NAMES[tgt]}. Note that you should only output the translated result "
                f"without any additional explanation:\n\n{text}")

    def one(self, text, tgt, max_new=64):
        r = self.m.create_chat_completion(messages=[{"role": "user", "content": self.prompt(text, tgt)}], temperature=0.0,
                                          top_k=1, repeat_penalty=1.05, max_tokens=max_new)
        c = r["choices"][0]; out = (c["message"]["content"] or "").strip().split("\n")[0].strip()
        return out, 0.5                            # llama.cpp gives no token log-probs without logits_all (2x slower)

    def translate(self, texts, src, tgt):
        return [self.one(t, tgt) for t in texts]
